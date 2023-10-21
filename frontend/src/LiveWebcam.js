import React from 'react';

const LiveWebcam = () => {
  const webcamStreamURL = 'https://www.earthcam.com/cams/includes/image.php?logo=0&amp;playbutton=0&amp;s=1&amp;img=0RC1tzOXJTEaJtMiEAhICw%3D%3D';

  return (
    <div>
      <video controls autoPlay src={webcamStreamURL} />
    </div>
  );
};

export default LiveWebcam;


