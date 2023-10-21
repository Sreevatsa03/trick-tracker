import React from 'react';

const TrickDetected = () => {

    let trick = "Kickflip"
    let confidence = .83 * 100
    let percent = confidence.toString() + "%"

    if (true) {
        return (
            <div>
              <h2>{trick} detected with {percent} accuracy!</h2>
            </div>
          );
    } else {
        return (<div></div>);
    }

  
};

export default TrickDetected;