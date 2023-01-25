from src.features.postprocessing import post_process_plate, post_process_plates


class TestsPostProcessing():
    def test_post_process_plate(self):

        plates = [
            {
                "plate": "*#xasads!",
                "result": "xasads"
            },
            {
                "plate": "FT123SA",
                "result": "FT123SA"
            },
            {
                "plate": "**##!!",
                "result": None
            },
            {
                "plate": "123132132132132132",
                "result": None
            },
            {
                "plate": "12",
                "result": None
            }
        ]

        for element in plates:
            plate = element["plate"]
            result = element["result"]

            assert post_process_plate(plate) == result

    def test_post_process_plates(self):

        plates = ["*#xasads!", "FT123SA", "**##!!", "123132132132132132", ""]
        result = ["xasads", "FT123SA"]

        assert post_process_plates(plates) == result