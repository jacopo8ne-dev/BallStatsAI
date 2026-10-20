"""Verifica la pipeline completa: YOLO simulato, ByteTrack e video realmente eseguiti."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from src.detection import rileva_persone, trova_classi_modello, separa_rilevamenti
from src.shots import HoopTracker


NAMES = {0:"Ref", 1:"Ball", 2:"Clock", 3:"Hoop", 4:"Overlay", 5:"Player", 6:"Scoreboard"}


def row(cls, bbox, score=0.9):
    return [*bbox, score, cls]


def run_pipeline(root, detections, count, skip=1, **options):
    """Mantiene reali codec e tracker, sostituendo soltanto le predizioni del modello."""
    source = root / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400,240))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((240,400,3), 10 + index % 20 * 5, dtype=np.uint8))
    writer.release()
    (root / "best.pt").touch()
    model = Mock()
    model.names = NAMES
    seen = []
    expected_indices = iter(range(0,count,skip))
    def predict(image, classes, **kwargs):
        index = next(expected_indices)  # Una chiamata sola per frame elaborato, mai una per oggetto.
        seen.append(index)
        assert set(classes) == ({1,5} if options.get("hoop_bbox") is not None else {1,3,5})
        expected_conf = min(options.get("ball_confidence",0.05), options.get("confidence",0.1),
                            options.get("hoop_confidence",0.3), 0.01 if index < 150 else 1)
        assert kwargs["conf"] == expected_conf
        boxes = Boxes(np.asarray(detections.get(index,[]),dtype=np.float32).reshape(-1,6), image.shape[:2])
        return [SimpleNamespace(boxes=boxes)]
    model.predict.side_effect = predict
    with patch("src.detection.YOLO", return_value=model) as load, patch("builtins.print") as messages:
        result = rileva_persone(source, root/"output.mp4", root/"best.pt", skip_frames=skip,
            tracks_path=root/"players.json", ball_tracks_path=root/"balls.json",
            possession_path=root/"possession.json", shot_path=root/"shots.json", debug_dir=root/"debug", **options)
    assert load.call_count == 1
    model.track.assert_not_called()
    assert seen == list(range(0,count,skip))
    records = {name:json.loads((root/f"{name}.json").read_text()) for name in ("players","balls","possession","shots")}
    capture = cv2.VideoCapture(str(root/"output.mp4"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    images = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        images.append(image)
    capture.release()
    debug = list((root/"debug").glob("*/frame_*.jpg"))
    assert len(debug) == min(count,150)
    assert all(cv2.imread(str(path)) is not None for path in debug)
    return result, records, images, fps, [call.args[0] for call in messages.call_args_list]


class TestModelloPersonalizzato(unittest.TestCase):
    def test_indici_dinamici(self):
        self.assertEqual(trova_classi_modello(NAMES), {"Player":5,"Ball":1,"Hoop":3})
        self.assertEqual(trova_classi_modello(["Ref","Player","Hoop","Ball"]), {"Player":1,"Ball":3,"Hoop":2})
        with self.assertRaises(ValueError):
            trova_classi_modello({0:"person",32:"sports ball"})

    def test_canestro_periodico_ritrovato_oltre_300_frame(self):
        tracker = HoopTracker()
        for index in range(330):
            self.assertIsNone(tracker.update([], index))
        current = tracker.update([row(3,[10,10,30,20],0.8)],330)
        self.assertEqual(current["frame"],330)
        self.assertEqual(tracker.update([],360),current)
        self.assertEqual(tracker.update([row(3,[20,10,40,20],0.9)],361),current)
        updated = tracker.update([row(3,[20,10,40,20],0.9)],390)
        self.assertEqual(updated["bbox"],(20,10,40,20))

    def test_canestro_soglia_e_manuale(self):
        tracker = HoopTracker(confidence=0.4)
        self.assertIsNone(tracker.update([row(3,[10,10,30,20],0.39)],0))
        self.assertIsNotNone(tracker.update([row(3,[10,10,30,20],0.4)],30))
        manual = HoopTracker(manual_bbox=(10,10,30,20))
        self.assertEqual(manual.update([row(3,[50,50,70,70])],30)["bbox"],(10,10,30,20))
        for invalid in (0,-1,1.1,float("nan"),float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HoopTracker(confidence=invalid)

    def test_boxes_assenti_e_coordinate_invalide(self):
        classes=trova_classi_modello(NAMES)
        for results in ([],[SimpleNamespace(boxes=None)]):
            self.assertEqual(separa_rilevamenti(results,classes,(240,400)),
                             {"Player":[],"Ball":[],"Hoop":[]})
        invalid=[row(1,[0,0,float("nan"),30]),row(1,[0,0,10,10],float("inf")),
                 row(1,[20,20,10,10]),row(3,[-1,10,30,20]),row(3,[10,10,500,20])]
        boxes=Boxes(np.asarray(invalid,dtype=np.float32),(240,400))
        self.assertEqual(separa_rilevamenti([SimpleNamespace(boxes=boxes)],classes,(240,400)),
                         {"Player":[],"Ball":[],"Hoop":[]})

    def test_inferenza_unica_classi_filtri_e_skip(self):
        detections = {i:[row(5,[20,130,50,200]), row(0,[80,130,110,200]),
                         row(2,[0,0,10,10]),row(4,[0,0,10,10]),row(6,[0,0,10,10]),
                         row(32,[150,150,160,160]),row(1,[0,0,300,200]),
                         row(1,[100+i,100,110+i,110],0.2)] for i in range(10)}
        with TemporaryDirectory() as directory:
            result, records, images, fps, _ = run_pipeline(Path(directory),detections,10,skip=2)
        self.assertEqual(result,(5,5))
        self.assertEqual(len(images),5)
        self.assertAlmostEqual(fps,15)
        self.assertEqual([r["frame"] for r in records["possession"]],[0,2,4,6,8])
        self.assertEqual(len({r["id"] for r in records["players"]}),1)
        self.assertEqual(len(records["balls"]),5)
        self.assertTrue(all(r["bbox"][2]-r["bbox"][0] < 20 for r in records["balls"]))
        self.assertEqual({r["source"] for r in records["balls"]},{"yolo"})

    def test_interpolazione_coerente_video_e_json_e_coda_vuota(self):
        detections = {0:[row(1,[100,100,110,110])],6:[row(1,[112,100,122,110])]}
        with TemporaryDirectory() as directory:
            _, records, images, _, _ = run_pipeline(Path(directory),detections,10)
        self.assertEqual([r["frame"] for r in records["balls"]],list(range(7)))
        self.assertEqual([r["source"] for r in records["balls"]], ["yolo"]+["interpolated"]*5+["yolo"])
        for index, record in enumerate(records["balls"]):
            self.assertAlmostEqual(record["bbox"][0],100+2*index)
        for index, image in enumerate(images):
            region = image[85:125,85:135]
            red = (region[:,:,2]>150) & (region[:,:,0]<100) & (region[:,:,1]<100)
            self.assertEqual(bool(red.sum()>10), index <= 6)
        self.assertTrue(all(r["player_id"] is None for r in records["possession"]))

    def test_debug_canestro_sotto_soglia_e_aggiornamento(self):
        detections = {0:[row(3,[90,90,110,100],0.1)],30:[row(3,[90,90,110,100],0.7)],
                      60:[row(3,[190,90,210,100],0.8)]}
        with TemporaryDirectory() as directory:
            root=Path(directory)
            with patch("src.detection.cv2.putText", wraps=cv2.putText) as labels:
                _, _, images, _, messages = run_pipeline(root,detections,92)
            texts=[call.args[1] for call in labels.call_args_list]
            self.assertTrue(any("Hoop 0.100 SOTTO SOGLIA" in text for text in texts))
            self.assertTrue(any("Riferimento frame 60 conf 0.800" in text for text in texts))
        self.assertTrue(any("trovato al frame 30" in message for message in messages))
        self.assertTrue(any("aggiornato al frame 60" in message for message in messages))
        for index in (0,29,30,59,60,90,91):
            x = 100 if index < 60 else 200
            region = images[index][75:115,x-20:x+20]
            green=(region[:,:,1]>160)&(region[:,:,0]<100)&(region[:,:,2]<100)
            self.assertEqual(bool(green.sum()>20),index>=30)

    def test_interpolazione_con_skip_mantiene_indici_originali(self):
        detections={0:[row(1,[100,100,110,110])],6:[row(1,[112,100,122,110])]}
        with TemporaryDirectory() as directory:
            _,records,images,_,_=run_pipeline(Path(directory),detections,8,skip=2)
        self.assertEqual([r["frame"] for r in records["balls"]],[0,2,4,6])
        self.assertEqual([r["bbox"][0] for r in records["balls"]],[100,104,108,112])
        self.assertEqual([r["source"] for r in records["balls"]],["yolo","interpolated","interpolated","yolo"])
        self.assertEqual(len(images),4)

    def test_manuale_senza_ricerca_e_video_vuoto(self):
        with TemporaryDirectory() as directory:
            _, records, images, _, _ = run_pipeline(Path(directory),{},6,hoop_bbox=(90,90,110,100))
        self.assertEqual(records["balls"],[])
        self.assertEqual(records["shots"],[])
        self.assertEqual(len(records["possession"]),6)
        for image in images:
            region=image[75:115,75:125]
            self.assertTrue(((region[:,:,1]>160)&(region[:,:,0]<100)&(region[:,:,2]<100)).any())

    def test_skip_non_divisore_30_e_debug_150(self):
        detections={35:[row(3,[90,90,110,100])]}
        with TemporaryDirectory() as directory:
            _, records, images, fps, messages=run_pipeline(Path(directory),detections,153,skip=7)
        self.assertEqual(len(images),22)
        self.assertEqual(len(records["possession"]),22)
        self.assertAlmostEqual(fps,30/7,places=2)
        self.assertTrue(any("trovato al frame 35" in message for message in messages))

    def test_fallback_spento_e_attivabile(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory, patch(
                "src.detection.detect_ball_by_color",return_value=[100,100,110,110]
            ) as color:
                _, records, _, _, _=run_pipeline(Path(directory),{},2,use_color_fallback=enabled)
                self.assertEqual(color.call_count,2 if enabled else 0)
                self.assertEqual(len(records["balls"]),2 if enabled else 0)
                self.assertTrue(all(r["source"]=="color" for r in records["balls"]))

    def test_cli_default_e_opzioni(self):
        from main import main
        for extra in ([],["--hoop-bbox","10,10,30,20","--skip-frames","2","--use-color-fallback"]):
            with self.subTest(extra=extra), patch("sys.argv",["main.py","--input","video.mp4",*extra]), patch(
                "main.rileva_persone", return_value=(0,0)
            ) as analyze, patch("builtins.print"):
                main()
            self.assertEqual(analyze.call_args.kwargs["ball_confidence"],0.05)
            self.assertEqual(analyze.call_args.kwargs["skip_frames"],2 if extra else 1)
            self.assertEqual(analyze.call_args.kwargs["hoop_bbox"],(10,10,30,20) if extra else None)
            self.assertEqual(analyze.call_args.kwargs["use_color_fallback"],bool(extra))
        for option,value in (("--skip-frames","0"),("--skip-frames","1.5"),("--hoop-bbox","1,1,0,0"),
                             ("--ball-confidence","nan"),("--hoop-confidence","inf")):
            with self.subTest(option=option,value=value), patch("sys.argv",["main.py","--input","v.mp4",option,value]), patch("sys.stderr"), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
