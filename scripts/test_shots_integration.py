"""Verifica tiri, video e JSON dopo l'inferenza unica e il tracking separato."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import cv2

from test_custom_model import run_pipeline, row


class TestTiriVideo(unittest.TestCase):
    def test_video_json_e_risultati(self):
        for made, points in [(True,[(100,150),(100,140),(100,110),(100,70),(100,80),(100,95),(100,105)]),
                             (False,[(100,150),(100,140),(120,70),(125,80),(130,105)])]:
            delay=60 if made else 0
            points=[points[0]]*delay+points
            detections={}
            for index,(x,y) in enumerate(points):
                detections[index]=[row(5,[20,130,40,170]),row(1,[x-2,y-2,x+2,y+2],.8)]
                if index>=delay:
                    detections[index].append(row(3,[90,90,110,100]))
            with self.subTest(made=made), TemporaryDirectory() as directory, patch(
                "src.detection.cv2.putText",wraps=cv2.putText
            ) as labels:
                _,records,images,_,_=run_pipeline(Path(directory),detections,len(points))
            player_id=records["players"][0]["id"]
            self.assertEqual(records["shots"],[{"frame":delay,"player_id":player_id,"made":made,
                "shot_x":100.0,"shot_y":150.0,"distance":55.0}])
            texts=[call.args[1] for call in labels.call_args_list]
            self.assertTrue(any("IN CORSO" in text for text in texts))
            self.assertTrue(any(("FATTO" if made else "SBAGLIATO") in text for text in texts))
            self.assertEqual(len(images),len(points))

    def test_tiro_con_skip_tempi_originali(self):
        points=[(100,150),(100,130),(100,110),(100,70),(100,80),(100,95),(100,105)]
        detections={2*i:[row(5,[20,130,40,170]),row(1,[x-2,y-2,x+2,y+2],.8)] for i,(x,y) in enumerate(points)}
        with TemporaryDirectory() as directory:
            _,records,_,_,_=run_pipeline(Path(directory),detections,14,skip=2,hoop_bbox=(90,90,110,100))
        self.assertEqual(len(records["shots"]),1)
        self.assertTrue(records["shots"][0]["made"])
        self.assertEqual(records["shots"][0]["frame"],0)


if __name__ == "__main__":
    unittest.main()
