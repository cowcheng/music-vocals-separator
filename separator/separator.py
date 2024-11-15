from separator import utils


class Separator:
    def pitch_fix(self, source, sr_pitched, org_mix):
        """
        Adjusts the pitch of the source audio and matches its shape to the original mix.

        This method shifts the pitch of the `source` audio by a specified number of semitones
        and ensures that the resulting audio array matches the shape of the original mix.

        Parameters:
            source (array-like): The audio data to be pitch-shifted.
            sr_pitched (int): The sample rate of the pitched source audio.
            org_mix (array-like): The original mix audio data used for shape matching.

        Returns:
            array-like: The pitch-corrected and shape-matched source audio data.
        """

        semitone_shift = self.semitone_shift
        source = utils.change_pitch_semitones(
            source, sr_pitched, semitone_shift=semitone_shift
        )[0]
        source = utils.match_array_shapes(source, org_mix)
        return source
