import re

def post_process_plates(plates):
    """
    Post processes a list of plates
    The post processing include:
    - removing special characters
    - removing plates which are too long
    - removing plates which are too short

    After the postprocessing, the plates which are left without any character are removed
    """

    def clean_prediction(plate):
        return re.sub(r'\W+', '', plate)

    def discard_plate_too_long(plate):
        return not len(plate) > 10

    def discard_plate_too_short(plate):
        return not len(plate) < 4

    def get_plates_with_most_votest(plates):

        if len(plates) == 0:
            return []

        votes = {}
        for plate in plates:
            if plate in votes:
                votes[plate] += 1
            else:
                votes[plate] = 1

        list_of_tuple = [(k, v) for k, v in votes.items()]
        ordered_votes = sorted(list_of_tuple, key=lambda x: x[1], reverse=1)
        plates_to_keep = [ordered_votes[0]]  # always take the first
        for (plate, votes) in ordered_votes[1:]:
            if votes == plates_to_keep[0][1]:
                plates_to_keep.append((plate, votes))

        plates_to_keep = [plate for (plate, votes) in plates_to_keep]
        return plates_to_keep

    plates = list(map(clean_prediction, plates))
    plates = list(filter(discard_plate_too_long, plates))
    plates = list(filter(discard_plate_too_short, plates))

    plates = get_plates_with_most_votest(plates)

    return plates

def post_process_plate(plate):    
    """
    Perform postprocessing on a plate calling the more general version of this method: post_process_plate
    """
    
    predictions = post_process_plates([plate])
    return predictions[0] if len(predictions) else None