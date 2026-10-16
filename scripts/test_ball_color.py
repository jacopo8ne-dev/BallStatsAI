"""Test del fallback HSV con immagini sintetiche, senza scaricare modelli."""

import unittest  # Usa il sistema di test incluso in Python.

import cv2  # Disegna le forme di prova e converte i colori.
import numpy as np  # Crea immagini vuote.

from src.ball_color import detect_ball_by_color, valida_range_hsv


def colore_bgr(h):
    # Sceglie un colore preciso in HSV e lo converte nel formato delle immagini OpenCV.
    return tuple(int(value) for value in cv2.cvtColor(np.uint8([[[h, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])


class TestColorePalla(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)  # Sfondo nero per isolare il colore cercato.

    def test_cerchio_arancione(self):
        cv2.circle(self.frame, (100, 90), 12, colore_bgr(10), -1)
        originale = self.frame.copy()
        bbox = detect_ball_by_color(self.frame)
        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertLessEqual(abs((x1+x2)/2 - 100), 1)
        self.assertLessEqual(abs((y1+y2)/2 - 90), 1)
        self.assertTrue(all(type(value) is int for value in bbox))
        np.testing.assert_array_equal(self.frame, originale)  # Il fallback non deve modificare l'immagine originale.

    def test_sfondo_e_rumore(self):
        self.assertIsNone(detect_ball_by_color(self.frame))
        self.frame[20, 20] = colore_bgr(10)
        self.assertIsNone(detect_ball_by_color(self.frame))

    def test_esclude_rettangoli_e_grandi_superfici(self):
        for angolo in [(60, 60), (150, 40), (290, 190)]:
            with self.subTest(angolo=angolo):
                frame = self.frame.copy()
                cv2.rectangle(frame, (20, 20), angolo, colore_bgr(10), -1)
                self.assertIsNone(detect_ball_by_color(frame))

    def test_range_personalizzato(self):
        cv2.circle(self.frame, (100, 90), 12, colore_bgr(25), -1)
        self.assertIsNone(detect_ball_by_color(self.frame))
        self.assertIsNotNone(detect_ball_by_color(self.frame, ((20, 100, 100), (30, 255, 255))))

    def test_sceglie_la_forma_piu_circolare(self):
        cv2.circle(self.frame, (60, 90), 12, colore_bgr(10), -1)
        cv2.ellipse(self.frame, (200, 90), (17, 12), 0, 0, 360, colore_bgr(10), -1)
        bbox = detect_ball_by_color(self.frame)
        self.assertLess(bbox[2], 100)

    def test_range_non_validi(self):
        for intervallo in [((15, 100, 100), (5, 255, 255)), ((0, 0, 0), (180, 255, 255)), ((0, -1, 0), (15, 255, 255)), ((5.5, 100, 100), (15, 255, 255)), ((5, 100), (15, 255))]:
            with self.subTest(intervallo=intervallo):
                with self.assertRaises(ValueError):
                    valida_range_hsv(intervallo)


if __name__ == "__main__":
    unittest.main()  # Riporta i risultati e termina con errore se una verifica fallisce.
