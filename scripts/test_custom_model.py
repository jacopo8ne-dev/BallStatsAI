"""Prova classi dinamiche, canestro automatico e frame senza oggetti con risultati controllati."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.detection import rileva_persone, trova_classi_modello
from src.shots import detect_hoop


def box(class_id, bbox, confidence=0.9, track_id=None):
    # Simula il formato delle Boxes Ultralytics senza inventare pesi addestrati.
    return SimpleNamespace(cls=np.array([class_id]), xyxy=np.array([bbox]),
                           conf=np.array([confidence]), id=np.array([track_id]) if track_id is not None else None)


class TestModelloPersonalizzato(unittest.TestCase):
    def test_indici_dinamici_da_dizionario_e_lista(self):
        self.assertEqual(trova_classi_modello({0:"Hoop", 4:"Player", 2:"Ball", 3:"Ref"}),
                         {"Player":4, "Ball":2, "Hoop":0})
        self.assertEqual(trova_classi_modello(["Ref", "Player", "Clock", "Hoop", "Ball"]),
                         {"Player":1, "Ball":4, "Hoop":3})
        with self.assertRaisesRegex(ValueError, "Player, Ball e Hoop"):
            trova_classi_modello({0:"person", 32:"sports ball"})

    def test_canestro_piu_sicuro_e_rilevamenti_assenti(self):
        model = Mock()
        model.predict.side_effect = [[], [SimpleNamespace(boxes=None)],
            [SimpleNamespace(boxes=[box(2,[90,90,110,100],0.6), box(0,[0,0,20,20],0.99)])],
            [SimpleNamespace(boxes=[box(2,[80,70,120,90],0.95), box(2,[10,10,10,20],0.99)])], []]
        frames = [np.zeros((200,200,3), dtype=np.uint8)] * 5
        self.assertEqual(detect_hoop(model, frames, 2), (80,70,120,90))
        self.assertTrue(all(call.kwargs["classes"] == [2] for call in model.predict.call_args_list))

    def run_video(self, empty=False):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"input.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (200,200))
            self.assertTrue(writer.isOpened())
            for i in range(13):  # Più di dieci frame: controlla anche il passaggio dal buffer al video.
                writer.write(np.full((200,200,3), 10+i*10, dtype=np.uint8))
            writer.release()
            (root/"best.pt").touch()  # File fittizio: YOLO è sostituito dai mock di questo test.
            model = Mock()
            model.names = {0:"Ref", 1:"Ball", 2:"Clock", 3:"Hoop", 4:"Overlay", 5:"Player", 6:"Scoreboard"}
            observed = []
            def track(frame, classes, **kwargs):
                self.assertEqual(classes, [5])
                observed.append(float(frame.mean()))
                if empty or len(observed) in (3,4):
                    return [SimpleNamespace(boxes=None if len(observed) == 3 else [])]
                # Anche risultati di classi estranee vengono ignorati prima dell'esportazione.
                return [SimpleNamespace(boxes=[box(5,[20,130,40,170],track_id=7),
                    box(0,[40,130,60,170],track_id=99), box(2,[0,0,10,10],track_id=90),
                    box(4,[0,0,10,10],track_id=91), box(6,[0,0,10,10],track_id=92)])]
            model.track.side_effect = track
            detector = Mock()
            hoop_calls = []
            ball_calls = []
            def predict(frame, classes, **kwargs):
                if classes == [3]:
                    hoop_calls.append(1)
                    # Il canestro compare solo nel nono frame e non è mai più rilevato.
                    boxes = [box(3,[90,90,110,100])] if len(hoop_calls) == 9 and not empty else []
                else:
                    self.assertEqual(classes, [1])
                    ball_calls.append(1)
                    boxes = [] if empty else [box(1,[30,140,34,144]), box(0,[50,50,55,55])]
                return [SimpleNamespace(boxes=boxes)]
            detector.predict.side_effect = predict
            with patch("src.detection.YOLO", side_effect=[model, detector]):
                count, _ = rileva_persone(source, root/"output.mp4", root/"best.pt",
                    tracks_path=root/"players.json", ball_tracks_path=root/"balls.json",
                    possession_path=root/"possession.json", shot_path=root/"shots.json", use_color_fallback=False)
            self.assertEqual(count, 13)
            self.assertEqual(len(hoop_calls), 10)
            self.assertEqual(len(ball_calls), 13)
            self.assertEqual(len(observed), 13)
            self.assertTrue(all(a < b for a,b in zip(observed, observed[1:])))
            records = {name:json.loads((root/f"{name}.json").read_text()) for name in ["players","balls","possession","shots"]}
            self.assertEqual([r["frame"] for r in records["possession"]], list(range(13)))
            self.assertEqual(records["shots"], [])
            if empty:
                self.assertEqual(records["players"], [])
                self.assertEqual(records["balls"], [])
                self.assertTrue(all(r["player_id"] is None for r in records["possession"]))
            else:
                self.assertEqual({r["id"] for r in records["players"]}, {7})
                self.assertEqual(len(records["players"]), 11)
                self.assertEqual(len(records["balls"]), 13)
            capture = cv2.VideoCapture(str(root/"output.mp4"))
            decoded = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                roi = image[75:115,75:125]
                green = (roi[:,:,1] > 160) & (roi[:,:,0] < 100) & (roi[:,:,2] < 100)
                self.assertEqual(bool(green.sum() > 20), not empty)  # Il cerchio resta anche nei frame senza classi.
                decoded += 1
            capture.release()
            self.assertEqual(decoded, 13)

    def test_canestro_tardivo_ignorando_arbitri_e_overlay(self):
        self.run_video()

    def test_video_senza_nessuna_classe(self):
        self.run_video(empty=True)

    def test_cli_modello_default_e_canestro_non_richiesto(self):
        from main import main
        with patch("sys.argv", ["main.py", "--input", "partita.mp4"]), patch("main.rileva_persone", return_value=(0,0)) as analyze:
            main()
        self.assertEqual(analyze.call_args.args[2], Path("models/best.pt"))
        self.assertNotIn("hoop_bbox", analyze.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
