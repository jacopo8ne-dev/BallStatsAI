"""Test del possesso: esegui python -m unittest discover -s scripts -p test_possession.py."""

import math  # Fornisce valori non validi per verificare il controllo della soglia.
import unittest  # Esegue i test senza installare altre librerie.

from src.possession import calcola_possesso  # Importa il calcolo indipendente da YOLO.


class TestPossesso(unittest.TestCase):
    def setUp(self):
        # Il primo giocatore ha centro (3, 4), il secondo (30, 40); la palla ha centro (0, 0).
        self.giocatori = [{"id": 7, "bbox": [2, 3, 4, 5]}, {"id": 9, "bbox": [29, 39, 31, 41]}]
        self.palle = [{"bbox": [-1, -1, 1, 1], "confidence": 0.8}]

    def test_sceglie_il_giocatore_piu_vicino(self):
        risultato = calcola_possesso(list(reversed(self.giocatori)), self.palle, 12, 150)
        self.assertEqual(risultato, {"frame": 12, "player_id": 7, "ball_distance": 5.0, "confidence": 0.8})

    def test_distanza_uguale_o_superiore_alla_soglia(self):
        for soglia in [5, 4]:  # Una distanza uguale alla soglia è esclusa.
            with self.subTest(soglia=soglia):
                risultato = calcola_possesso(self.giocatori, self.palle, 0, soglia)
                self.assertIsNone(risultato["player_id"])
                self.assertEqual(risultato["ball_distance"], 5.0)
                self.assertEqual(risultato["confidence"], 0.8)

    def test_senza_palla_non_conserva_il_possesso(self):
        calcola_possesso(self.giocatori, self.palle, 0)
        self.assertEqual(calcola_possesso(self.giocatori, [], 1),
                         {"frame": 1, "player_id": None, "ball_distance": -1.0, "confidence": 0.0})

    def test_senza_giocatori(self):
        self.assertEqual(calcola_possesso([], self.palle, 2),
                         {"frame": 2, "player_id": None, "ball_distance": -1.0, "confidence": 0.8})

    def test_sceglie_la_palla_piu_sicura(self):
        # La palla più sicura è vicina al secondo giocatore: non sceglie quella più vicina al primo.
        palle = self.palle + [{"bbox": [29, 39, 31, 41], "confidence": 0.9}]
        risultato = calcola_possesso(self.giocatori, palle, 0)
        self.assertEqual(risultato["player_id"], 9)
        self.assertEqual(risultato["ball_distance"], 0.0)
        self.assertEqual(risultato["confidence"], 0.9)

    def test_centri_con_mezzi_pixel(self):
        giocatori = [{"id": 1, "bbox": [0, 0, 1, 1]}]
        risultato = calcola_possesso(giocatori, self.palle, 0)
        self.assertAlmostEqual(risultato["ball_distance"], math.sqrt(0.5))

    def test_parita_di_distanza(self):
        giocatori = self.giocatori[:1] + [{"id": 11, "bbox": [-4, -5, -2, -3]}]
        self.assertEqual(calcola_possesso(giocatori, self.palle, 0)["player_id"], 7)

    def test_soglie_non_valide(self):
        for soglia in [0, -1, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(soglia=soglia):
                with self.assertRaises(ValueError):
                    calcola_possesso(self.giocatori, self.palle, 0, soglia)


if __name__ == "__main__":
    unittest.main()  # Mostra quanti test sono passati e segnala eventuali errori.
