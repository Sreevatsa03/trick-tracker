import React from 'react';

const LiveWebcam = () => {
  const videoStreamURL = 'http://127.0.0.1:5000/video_feed';  // Update this URL to match your Flask route

  return (
    <div>
      <h1>Live Video Stream</h1>
      <img src={videoStreamURL} alt="webcam" />
    </div>
  );
};

export default LiveWebcam;



