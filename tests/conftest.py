import cv2
import pytest


@pytest.fixture(scope="session")
def face_video_bytes(tmp_path_factory):
    pytest.importorskip("skimage")
    from skimage import data

    path = tmp_path_factory.mktemp("vid") / "face.avi"
    frame = cv2.resize(data.astronaut()[:, :, ::-1].copy(), (640, 480))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    for _ in range(20):
        writer.write(frame)
    writer.release()
    return path.read_bytes()
