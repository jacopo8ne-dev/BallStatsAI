"""Prove del tracker della palla: Kalman, memoria dell'ID, stima, punti statici e interpolazione."""

import unittest
from src.ball_tracking import BallTracker, KalmanPalla, forma_da_palla, interpola_palla, vicino_a_giocatore


def candidate(x, confidence=0.2, y=100):
    return [x,y,x+12,y+12,confidence,0]


class TestBallTracking(unittest.TestCase):
    def test_inizio_forte_o_due_deboli_consecutivi(self):
        self.assertIsNone(BallTracker().update([candidate(100,.06)],0))
        first=BallTracker().update([candidate(100)],0)
        self.assertEqual(first["id"],1)
        tracker=BallTracker()
        self.assertIsNone(tracker.update([candidate(100,.06)],0))
        confirmed=tracker.update([candidate(104,.06)],1)
        self.assertIsNotNone(confirmed)  # Due deboli vicini in frame consecutivi bastano.
        tracker=BallTracker()
        tracker.update([candidate(100,.06)],0)
        self.assertIsNone(tracker.update([candidate(400,.06)],1))

    def test_palla_veloce_senza_sovrapposizione(self):
        tracker=BallTracker()
        first=tracker.update([candidate(100)],0)
        for frame in range(1,6):
            ball=tracker.update([candidate(100+25*frame,.06)],frame)  # 25 px per frame: nessuna sovrapposizione.
            self.assertIsNotNone(ball)
            self.assertEqual(ball["id"],first["id"])
        # La previsione segue la velocità: dopo 3 frame persi la palla è attesa intorno a x=300.
        self.assertTrue(tracker.movimento_valido(candidate(300)[:4],8))
        self.assertFalse(tracker.movimento_valido(candidate(100)[:4],8))

    def test_kalman_velocita_e_accelerazione(self):
        kalman=KalmanPalla(0,0)
        for frame in range(1,15):
            kalman.predict(1)
            kalman.update(10*frame,frame*frame)  # x a velocità costante, y accelerata (a=2).
        kalman.predict(1)
        x,y=kalman.posizione
        self.assertAlmostEqual(x,150,delta=3)
        self.assertAlmostEqual(y,225,delta=6)
        self.assertLess(kalman.mahalanobis2(150,225),kalman.mahalanobis2(250,225))

    def test_candidato_lontano_debole_ignorato_forte_stesso_id(self):
        tracker=BallTracker()
        first=tracker.update([candidate(100)],0)
        self.assertIsNone(tracker.update([candidate(600,.1)],1))
        jumped=tracker.update([candidate(600,.5)],2)
        self.assertEqual(jumped["bbox"][0],600)  # Una palla sicura lontana viene accettata...
        self.assertEqual(first["id"],jumped["id"])  # ...ed è la stessa palla: stesso id.

    def test_stima_trenta_frame_e_memoria_150(self):
        tracker=BallTracker(fps=30)
        for frame in range(5):
            first=tracker.update([candidate(100+10*frame)],frame)
        guess=tracker.stima(5,(600,1000))
        self.assertEqual((guess["source"],guess["estimated"],guess["id"]),("predicted",True,first["id"]))
        self.assertGreater(guess["bbox"][0],first["bbox"][0])  # Prosegue nella direzione del moto.
        for frame in range(5,34):
            self.assertIsNone(tracker.update([],frame))
        self.assertIsNotNone(tracker.stima(34,(600,1000)))  # 30 frame dopo l'ultima osservazione.
        self.assertIsNone(tracker.stima(35,(600,1000)))  # Oltre 30 frame non si disegna più...
        self.assertTrue(tracker.attivo(154))  # ...ma l'id resta in memoria per 150 frame.
        recovered=tracker.update([candidate(700)],150)
        self.assertEqual(recovered["id"],first["id"])  # Ri-associata dopo 5 secondi di occlusione.
        self.assertFalse(tracker.attivo(301))
        renewed=tracker.update([candidate(700)],301)
        self.assertNotEqual(renewed["id"],first["id"])

    def test_stima_dentro_immagine(self):
        tracker=BallTracker()
        tracker.update([candidate(950)],0)
        tracker.update([candidate(975,.06)],1)
        guess=tracker.stima(10,(600,1000))
        self.assertLessEqual(guess["bbox"][2],1000)

    def test_punto_statico_ignorato(self):
        tracker=BallTracker(fps=10)  # Statico dopo 10 osservazioni distribuite su almeno 1 secondo (10 frame).
        for frame in range(0,20,2):
            tracker.update([candidate(500,.3)],frame)
        self.assertTrue(tracker.static.statico(candidate(505)[:4]))
        self.assertFalse(tracker.static.statico(candidate(530)[:4]))
        self.assertIsNone(tracker.update([candidate(501,.9)],20))
        moved=tracker.update([candidate(501,.9),candidate(800,.3)],21)
        self.assertEqual(moved["bbox"][0],800)

    def test_inizio_solo_vicino_a_un_giocatore(self):
        players=[[90,40,130,120],[400,40,440,120],[600,40,640,120],[0,300,250,600]]  # L'ultimo è pubblico in primo piano.
        self.assertTrue(vicino_a_giocatore(candidate(190)[:4],players))
        self.assertFalse(vicino_a_giocatore(candidate(250)[:4],players))
        self.assertFalse(vicino_a_giocatore(candidate(100,y=400)[:4],players))
        tracker=BallTracker()
        self.assertIsNone(tracker.update([candidate(250,.5)],0,players=players))
        first=tracker.update([candidate(190,.5)],1,players=players)
        self.assertIsNotNone(first)
        # Una traccia attiva segue la palla anche lontano dai giocatori, per esempio durante un tiro.
        self.assertIsNotNone(tracker.update([candidate(230,.06)],2,players=players))

    def test_box_troppo_grandi_scartati(self):
        players=[[0,0,20,80],[50,0,70,80],[100,0,120,80]]
        self.assertTrue(forma_da_palla([0,0,10,20],players))  # Sfocata dal palleggio: alta e stretta va bene.
        self.assertFalse(forma_da_palla([0,0,20,30],players))  # Testa del pubblico: oltre 0.3 x 80 pixel.
        self.assertTrue(forma_da_palla([0,0,20,30],[]))

    def test_skip_frames(self):
        tracker=BallTracker(skip_frames=2)
        tracker.update([candidate(100,.06)],0)
        self.assertIsNotNone(tracker.update([candidate(110,.06)],2))

    def test_buchi_da_uno_a_ventinove(self):
        for missing in (1,5,29,30):
            tracker=BallTracker()
            first=tracker.update([candidate(100)],0)
            for frame in range(1,missing+1):
                tracker.update([],frame)
            last=tracker.update([candidate(112)],missing+1)
            filled=interpola_palla(first,last,range(missing+3))
            self.assertEqual(len(filled),missing if missing<=29 else 0)
            if filled:
                self.assertAlmostEqual(filled[1]["bbox"][0],100+12/(missing+1))
                self.assertEqual((filled[1]["source"],filled[1]["estimated"]),("interpolated",True))
            self.assertFalse(first["estimated"])
            self.assertEqual(interpola_palla(first,None,[1]),{})
            self.assertEqual(interpola_palla(None,last,[1]),{})


if __name__ == "__main__":
    unittest.main()
