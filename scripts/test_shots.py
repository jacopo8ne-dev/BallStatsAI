"""Traiettorie controllate per provare il rilevamento dei tiri senza YOLO."""

import unittest  # Esegue i test senza nuove dipendenze.

from src.shots import ShotDetector, parse_hoop_bbox


def ball(x, y):
    return [{"bbox": [x - 2, y - 2, x + 2, y + 2], "confidence": 0.8}]


class TestTiri(unittest.TestCase):
    def setUp(self):
        self.detector = ShotDetector((90, 90, 110, 100), 30)

    def sequence(self, points, owners=None):
        # Solo il primo frame ha il possesso: il tiratore va ricordato durante il rilascio.
        owners = owners if owners is not None else [7] + [None] * (len(points) - 1)
        labels = []
        for frame, (point, owner) in enumerate(zip(points, owners)):
            labels.append(self.detector.update(frame, ball(*point) if point else [], {"player_id": owner}))
        return labels

    def test_fatto_e_singolo_evento(self):
        labels = self.sequence([(100,150), (100,140), (100,110), (100,70), (100,80), (100,95), (100,105), (100,120)])
        self.assertIn("IN CORSO", labels[1])
        self.assertIn("FATTO", labels[6])
        self.assertEqual(self.detector.shots, [{"frame":0,"player_id":7,"made":True,"shot_x":100.0,"shot_y":150.0,"distance":55.0}])

    def test_sbagliato_lateralmente(self):
        self.sequence([(100,150), (100,140), (120,70), (125,80), (130,105)])
        self.assertEqual(len(self.detector.shots), 1)
        self.assertFalse(self.detector.shots[0]["made"])

    def test_tiro_corto(self):
        self.sequence([(100,150), (100,140), (100,130), (100,135)])
        self.assertFalse(self.detector.shots[0]["made"])

    def test_segmento_salta_tutto_il_rettangolo(self):
        self.sequence([(100,150), (100,140), (100,70), (100,120)])
        self.assertTrue(self.detector.shots[0]["made"])

    def test_ingresso_laterale_non_e_canestro(self):
        self.sequence([(100,150), (100,140), (60,70), (80,95), (100,105)])
        self.assertFalse(self.detector.shots[0]["made"])

    def test_niente_tiro_senza_possessore(self):
        self.sequence([(100,150), (100,130), (100,70), (100,120)], [None]*4)
        self.assertEqual(self.detector.shots, [])
        self.assertIsNone(self.detector.active)

    def test_salita_lenta_o_direzione_opposta(self):
        for points in [[(100,150),(100,148)], [(150,150),(170,140)]]:
            with self.subTest(points=points):
                self.setUp()
                self.sequence(points)
                self.assertIsNone(self.detector.active)

    def test_palla_persa_non_diventa_sbagliato(self):
        self.sequence([(100,150), (100,140), (100,70), None, (100,95), (100,105)])
        self.assertEqual(self.detector.shots, [])
        self.assertEqual(self.detector.unresolved, 1)

    def test_occlusione_prima_del_ferro_poi_attraversamento_visibile(self):
        self.sequence([(100,150), (100,140), None, (100,70), (100,80), (100,95), (100,105)])
        self.assertTrue(self.detector.shots[0]["made"])

    def test_due_tiri_distinti_con_nuovo_possesso(self):
        self.sequence([(100,150), (100,140), (100,110), (100,70), (100,80), (100,95), (100,105)])
        for frame in range(7, 23):  # Attende il recupero della palla e un nuovo possesso dopo la pausa.
            self.detector.update(frame, ball(100,150), {"player_id":9})
        for frame, point in enumerate([(100,140), (100,110), (100,70), (100,80), (100,95), (100,105)], start=23):
            self.detector.update(frame, ball(*point), {"player_id":None})
        self.assertEqual(len(self.detector.shots), 2)
        self.assertEqual([shot["player_id"] for shot in self.detector.shots], [7,9])
        self.assertEqual(self.detector.shots[1]["frame"], 22)

    def test_fine_video_e_timeout(self):
        self.sequence([(100,150), (100,140)])
        self.detector.finish_video(2)
        self.assertEqual(self.detector.shots, [])
        self.assertEqual(self.detector.unresolved, 1)
        self.setUp()
        self.sequence([(100,150), (100,140)])
        self.detector.update(40, [], {"player_id":None})
        self.assertEqual(self.detector.unresolved, 1)

    def test_niente_velocita_attraverso_frame_mancanti(self):
        self.sequence([(100,150), None, (100,120)], [7,None,None])
        self.assertIsNone(self.detector.active)

    def test_parametri_non_validi(self):
        for value in ["1,2,3", "3,2,1,4", "-1,2,3,4", "1,2,nan,4", "1.5,2,3,4"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_hoop_bbox(value)
        for speed in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                ShotDetector((90,90,110,100), 30, speed)


if __name__ == "__main__":
    unittest.main()
