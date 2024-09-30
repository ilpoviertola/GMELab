from pathlib import Path

import pytest

from data.evaluation_video import EvaluationVideo, EvaluationVideoDirectory, hash_string
from eval_utils.test_utils import gt_file_1, sample_file_1, sample_file_2
from eval_utils.exceptions import ConfigurationError
from eval_utils.file_utils import copy_file


def test_init_evaluation_video(gt_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(gt_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=True,
        extract_audio=True,
    )
    assert ev.video_file_path.as_posix() == gt_file_1
    assert ev.vfps == 25
    assert ev.afps == 24000
    assert ev.vcodec == "h264"
    assert ev.acodec == "aac"
    assert ev.is_ground_truth == True
    assert ev.audio_file_path == Path(gt_file_1).with_suffix(".wav")
    assert ev.start_time == None
    assert ev.end_time == None
    assert ev.duration == None
    assert ev.gt_evaluation_video_object == None
    assert ev.id == hash_string(ev.video_file_path.name)


def test_add_evaluation_video_to_directory(gt_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(gt_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)

    assert len(evd) == 1
    assert evd.video_variations[ev.id][0].video_file_path.as_posix() == gt_file_1


def test_delete_evaluation_video_from_directory(sample_file_1):
    sample_file_1_copy = Path(sample_file_1).with_name("copy.mp4")
    copy_file(sample_file_1, sample_file_1_copy)
    assert sample_file_1_copy.exists()
    ev = EvaluationVideo(
        video_file_path=sample_file_1_copy,
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=False,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1
    evd.remove_videos_by_name(sample_file_1_copy.name, delete_files=True)
    assert len(evd) == 0
    assert not sample_file_1_copy.exists()


def test_delete_gt_evaluation_video_from_directory(gt_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(gt_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=True,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1
    evd.remove_videos_by_name(ev.video_file_path.name, delete_files=True)
    assert len(evd) == 0
    assert Path(gt_file_1).exists()  # ground truth file should not be deleted


def test_add_two_variations_to_directory(gt_file_1, sample_file_1):
    ev_1 = EvaluationVideo(
        video_file_path=Path(gt_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=True,
    )
    ev_2 = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        gt_evaluation_video_object=ev_1,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_videos([ev_1, ev_2])

    assert len(evd) == 1
    assert len(evd.video_variations[ev_1.id]) == 2


def test_add_two_different_videos_to_directory(sample_file_1, sample_file_2):
    ev_1 = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
    )
    ev_2 = EvaluationVideo(
        video_file_path=Path(sample_file_2),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_videos([ev_1, ev_2])

    assert len(evd) == 2
    assert len(evd.video_variations[ev_1.id]) == 1
    assert len(evd.video_variations[ev_2.id]) == 1


def test_add_video_from_path_to_directory(sample_file_1):
    evd = EvaluationVideoDirectory()
    evd.add_video_from_path(
        video_file_path=sample_file_1,
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
    )

    assert len(evd) == 1
    assert len(evd.video_variations[hash_string(Path(sample_file_1).name)]) == 1


def test_add_non_existing_video_file_to_directory():
    evd = EvaluationVideoDirectory()
    with pytest.raises(ConfigurationError):
        evd.add_video_from_path(
            video_file_path="non_existing_file.mp4",
            vfps=25,
            afps=24000,
            vcodec="h264",
            acodec="aac",
            is_ground_truth=False,
        )


def test_create_new_variation(sample_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1

    evd.create_new_variation(
        id=ev.id,
        vfps=5,
        extract_audio=True,
    )
    assert len(evd.video_variations[ev.id]) == 2
    assert evd.video_variations[ev.id][1].vfps == 5
    assert evd.video_variations[ev.id][1].audio_file_path == evd.video_variations[
        ev.id
    ][1].video_file_path.with_suffix(".wav")
    evd.remove_all_videos(delete_files=True)


def test_create_new_variations(sample_file_1, sample_file_2):
    ev_1 = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    ev_2 = EvaluationVideo(
        video_file_path=Path(sample_file_2),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_videos([ev_1, ev_2])
    assert len(evd) == 2

    evd.create_new_variatons(
        ids=[ev_1.id, ev_2.id],
        vfps=5,
        extract_audio=True,
    )
    assert len(evd) == 2
    assert len(evd.video_variations[ev_1.id]) == 2
    assert len(evd.video_variations[ev_2.id]) == 2
    evd.remove_all_videos(delete_files=True)


def test_create_new_variations_parallel(sample_file_1, sample_file_2):
    ev_1 = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    ev_2 = EvaluationVideo(
        video_file_path=Path(sample_file_2),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_videos([ev_1, ev_2])
    assert len(evd) == 2

    evd.create_new_variatons(
        ids=[ev_1.id, ev_2.id],
        vfps=5,
        extract_audio=True,
        parallel=True,
    )
    assert len(evd) == 2
    assert len(evd.video_variations[ev_1.id]) == 2
    assert len(evd.video_variations[ev_2.id]) == 2
    evd.remove_all_videos(delete_files=True)


def test_find_variation(sample_file_1, gt_file_1):
    ev_gt = EvaluationVideo(
        video_file_path=Path(gt_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=True,
        is_original_file=True,
        extract_audio=True,
    )
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
        gt_evaluation_video_object=ev_gt,
        extract_audio=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_videos([ev_gt, ev])
    assert len(evd) == 1

    variation = evd._find_variation()
    assert len(variation) == 2

    evd.create_new_variation(
        id=ev.id,
        vfps=5,
        extract_audio=True,
    )
    assert len(evd.video_variations[ev.id]) == 3
    variation = evd._find_variation(vfps=5, has_audio=True)
    assert len(variation) == 1
    assert variation[0].vfps == 5
    evd.remove_all_videos(delete_files=True)


def test_find_non_existing_variation(sample_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1

    variation = evd._find_variation(vfps=5)
    assert len(variation) == 0


def test_get_existing_path_to_directory_with_specs(sample_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1

    path = evd.get_path_to_directory_with_specs(vfps=25)
    assert path == ev.video_file_path.parent


def test_get_path_to_directory_with_new_specs(sample_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1

    path = evd.get_path_to_directory_with_specs(vfps=5)
    assert "video_h264_5fps_256side_audio_24000hz_aac" in path.as_posix()
    evd.remove_all_videos(delete_files=True)


def test_load_from_directory(sample_file_1):
    evd = EvaluationVideoDirectory()
    evd.load_from_directory(
        Path(sample_file_1).parent,
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
    )

    assert len(evd) == 1457


def test_find_variations_and_extract_audio(sample_file_1):
    ev = EvaluationVideo(
        video_file_path=Path(sample_file_1),
        vfps=25,
        afps=24000,
        vcodec="h264",
        acodec="aac",
        is_ground_truth=False,
        is_original_file=True,
    )
    evd = EvaluationVideoDirectory()
    evd.add_evaluation_video(ev)
    assert len(evd) == 1
    assert not ev.has_audio

    evd._find_variation(vfps=25, has_audio=True)
    assert ev.has_audio
