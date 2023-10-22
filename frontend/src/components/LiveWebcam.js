import "./styles.css";
import {
  useRecordWebcam,
  CAMERA_STATUS
} from "react-record-webcam";
import TrickDetected from "./TrickDetected";
import { useState } from "react";

const OPTIONS = {
  filename: "my-video",
  fileType: "mp4",
  width: 1920,
  height: 1080
};



export default function LiveWebcam() {
  const recordWebcam = useRecordWebcam(OPTIONS);
  const [isHidden, setIsHidden] = useState(true);

  const sendVideoToBackend = (blob) => {
    const formData = new FormData();
    formData.append('video', blob);

    fetch('http://127.0.0.1:8000/trick', {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        if (response.status === 200) {
          console.log('Video uploaded successfully');
        } else {
          console.error('Error uploading video');
        }
      })
      .catch((error) => {
        console.error('Error uploading video:', error);
      });

  };

  const exportVid = async () => {
    const blob = await recordWebcam.getRecording();
    sendVideoToBackend(blob);
    setIsHidden(false);
  };

  return (
    <><div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
      <h1>Record your trick:</h1>
      <p>Camera status: {recordWebcam.status}</p>
      <div>
        <button
          disabled={recordWebcam.status === CAMERA_STATUS.OPEN ||
            recordWebcam.status === CAMERA_STATUS.RECORDING ||
            recordWebcam.status === CAMERA_STATUS.PREVIEW}
          onClick={recordWebcam.open}
        >
          Open camera
        </button>
        <button
          disabled={recordWebcam.status === CAMERA_STATUS.CLOSED ||
            recordWebcam.status === CAMERA_STATUS.PREVIEW}
          onClick={recordWebcam.close}
        >
          Close camera
        </button>
        <button
          disabled={recordWebcam.status === CAMERA_STATUS.CLOSED ||
            recordWebcam.status === CAMERA_STATUS.RECORDING ||
            recordWebcam.status === CAMERA_STATUS.PREVIEW}
          onClick={recordWebcam.start}
        >
          Start recording
        </button>
        <button
          disabled={recordWebcam.status !== CAMERA_STATUS.RECORDING}
          onClick={recordWebcam.stop}
        >
          Stop recording
        </button>
        <button
          disabled={recordWebcam.status !== CAMERA_STATUS.PREVIEW}
          onClick={recordWebcam.retake}
        >
          Retake
        </button>
        <button
          disabled={recordWebcam.status !== CAMERA_STATUS.PREVIEW}
          onClick={recordWebcam.download}
        >
          Download
        </button>

      </div>

      <video
        ref={recordWebcam.webcamRef}
        style={{
          display: `${recordWebcam.status === CAMERA_STATUS.OPEN ||
              recordWebcam.status === CAMERA_STATUS.RECORDING
              ? "block"
              : "none"}`
        }}
        autoPlay
        muted />
      <video
        ref={recordWebcam.previewRef}
        style={{
          display: `${recordWebcam.status === CAMERA_STATUS.PREVIEW ? "block" : "none"}`
        }}
        controls />
      <button
        hidden={recordWebcam.status !== CAMERA_STATUS.PREVIEW}
        onClick={exportVid}
      >
        Ready for analysis!
      </button>
    </div><div>
        {!isHidden ? <TrickDetected /> : null}
      </div></>
  );
}




