import React from 'react';

const LiveWebcam = () => {
  const videoStreamURL = 'https://i0.wp.com/www.printmag.com/wp-content/uploads/2021/02/4cbe8d_f1ed2800a49649848102c68fc5a66e53mv2.gif?fit=476%2C280&ssl=1';  // Update this URL to match your Flask route

  return (
    <div>
      <h2>Live Webcam Footage:</h2>
      <img src={videoStreamURL} alt="flask server not running" />
    </div>
  );
};

export default LiveWebcam;



