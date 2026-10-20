"""Verifica video, JSON e annotazioni dei tiri con tracce YOLO controllate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.detection import rileva_persone


class TestTiriVideo(unittest.TestCase):
    def test_video_json_e_risultati(self):
        # Le due traiettorie hanno lo stesso tiratore: una attraversa il ferro, l'altra passa a lato.
        for made, points in [(True, [(100,150),(100,140),(100,110),(100,70),(100,80),(100,95),(100,105)]),
                             (False, [(100,150),(100,140),(120,70),(125,80),(130,105)])]:
            delay = 50 if made else 0  # Verifica anche un tiro dopo la scoperta tardiva del canestro.
            points = [points[0]] * delay + points
            with self.subTest(made=made), TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "input.mp4"
                writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (200,200))
                self.assertTrue(writer.isOpened())
                for _ in points:
                    writer.write(np.zeros((200,200,3), dtype=np.uint8))
                writer.release()
                (root/"model.pt").touch()  # Il caricamento dei pesi viene simulato soltanto in questo test.
                player = SimpleNamespace(id=np.array([7]), cls=np.array([5]), xyxy=np.array([[20,130,40,170]]))
                model = Mock()
                model.names = {0:"Ref", 1:"Ball", 2:"Clock", 3:"Hoop", 4:"Overlay", 5:"Player", 6:"Scoreboard"}
                model.track.return_value = [SimpleNamespace(boxes=[player])]
                ball_model = Mock()
                hoop_calls = []
                ball_results = iter([[SimpleNamespace(boxes=[SimpleNamespace(
                    xyxy=np.array([[x-2,y-2,x+2,y+2]]), cls=np.array([1]), conf=np.array([0.8]))])] for x,y in points])
                def predict(frame, classes, **kwargs):
                    if classes == [3]:  # Il canestro viene ora individuato automaticamente.
                        hoop_calls.append(1)
                        if len(hoop_calls) <= delay:
                            return [SimpleNamespace(boxes=[])]
                        return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=np.array([[90,90,110,100]]), cls=np.array([3]), conf=np.array([0.9]))])]
                    self.assertEqual(classes, [1])
                    return next(ball_results)
                ball_model.predict.side_effect = predict
                with patch("src.detection.YOLO", side_effect=[model, ball_model]), patch("src.detection.cv2.putText", wraps=cv2.putText) as labels:
                    count, _ = rileva_persone(source, root/"output.mp4", root/"model.pt",
                        tracks_path=root/"players.json", ball_tracks_path=root/"balls.json",
                        possession_path=root/"possession.json", shot_path=root/"shots.json")
                events = json.loads((root/"shots.json").read_text())
                self.assertEqual(count, len(points))
                self.assertTrue(all(call.kwargs["classes"] == [5] for call in model.track.call_args_list))
                self.assertEqual(len(hoop_calls), delay+1)
                self.assertEqual(events, [{"frame":delay,"player_id":7,"made":made,"shot_x":100.0,"shot_y":150.0,"distance":55.0}])
                texts = [call.args[1] for call in labels.call_args_list]
                self.assertTrue(any("TIRO!" in text and "IN CORSO" in text for text in texts))
                self.assertTrue(any(("FATTO" if made else "SBAGLIATO") in text for text in texts))
                capture = cv2.VideoCapture(str(root/"output.mp4"))
                decoded = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape[:2], (200,200))
                    # Il cerchio compare dalla scoperta del canestro e resta dopo il tiro.
                    roi = frame[75:115,75:125]
                    green = (roi[:,:,1] > 160) & (roi[:,:,0] < 100) & (roi[:,:,2] < 100)
                    self.assertEqual(bool(green.sum() > 20), decoded >= delay)
                    decoded += 1
                capture.release()
                self.assertEqual(decoded, len(points))


if __name__ == "__main__":
    unittest.main()
