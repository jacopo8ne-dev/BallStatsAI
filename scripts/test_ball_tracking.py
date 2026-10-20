"""Prove delle soglie ByteTrack reali, limiti di movimento e interpolazione."""

import unittest
from src.ball_tracking import BallTracker, interpola_palla


def candidate(x, confidence=0.2):
    return [x,100,x+20,120,confidence,1]


class TestBallTracking(unittest.TestCase):
    def test_soglie_e_recupero_debole(self):
        tracker=BallTracker()
        args=tracker.tracker.args
        self.assertEqual((args.track_high_thresh,args.track_low_thresh,args.track_buffer),(.15,.05,30))
        first=tracker.update([candidate(100)],0,(600,1000))
        second=tracker.update([candidate(101,.06)],1,(600,1000))
        self.assertIsNotNone(second)
        self.assertEqual(first["id"],second["id"])
        self.assertEqual(first["tracker_id"],second["tracker_id"])
        self.assertIsNone(BallTracker().update([candidate(100,.06)],0,(600,1000)))
        self.assertEqual(BallTracker(skip_frames=2).tracker.max_time_lost,15)

    def test_teletrasporto_e_soglia_200(self):
        tracker=BallTracker()
        first=tracker.update([candidate(100)],0,(600,1000))
        self.assertIsNone(tracker.update([candidate(301)],1,(600,1000)))
        self.assertEqual(tracker.last,first)
        second=tracker.update([candidate(300)],1,(600,1000))
        self.assertIsNotNone(second)  # A 200 pixel il candidato è ammesso, anche senza sovrapposizione.
        self.assertEqual(first["id"],second["id"])

    def test_perdita_lunga_limite_300_e_scadenza_id(self):
        tracker=BallTracker()
        first=tracker.update([candidate(100)],0,(600,1000))
        for frame in range(1,8):
            self.assertIsNone(tracker.update([],frame,(600,1000)))
        self.assertIsNone(tracker.update([candidate(401)],8,(600,1000)))
        recovered=tracker.update([candidate(400)],9,(600,1000))
        self.assertEqual(first["id"],recovered["id"])
        for frame in range(10,41):
            tracker.update([],frame,(600,1000))
        expired=tracker.update([candidate(400)],41,(600,1000))
        self.assertNotEqual(first["id"],expired["id"])
        self.assertEqual(interpola_palla(first,recovered,range(1,9)),{})

    def test_buchi_da_uno_a_cinque_non_estrapolati(self):
        for missing in range(1,7):
            tracker=BallTracker()
            first=tracker.update([candidate(100)],0,(600,1000))
            for frame in range(1,missing+1):
                tracker.update([],frame,(600,1000))
            last=tracker.update([candidate(112)],missing+1,(600,1000))
            filled=interpola_palla(first,last,range(missing+3))
            self.assertEqual(len(filled),missing if missing<=5 else 0)
            if filled:
                self.assertAlmostEqual(filled[1]["bbox"][0],100+12/(missing+1))
                self.assertEqual(filled[1]["source"],"interpolated")
            self.assertEqual(interpola_palla(first,None,[1]),{})
            self.assertEqual(interpola_palla(None,last,[1]),{})


if __name__ == "__main__":
    unittest.main()
