import React, { useEffect } from "react";
import { useState } from "react";
import React from 'react';
import axios from 'axios';


const TrickDetected = () => {

    const [trick, setTrick] = useState({})
    const [conf, setConf] = useState({})

    useEffect(() => {
      // Make a GET request to the Flask API
      axios.get('/trick?Accuracy=${conf}&Prediction=${trick}')
        .then((response) => {
          setTrick(response.trick);
          setConf(response.conf);
        })
        .catch((error) => {
          console.error('Error fetching data:', error);
        });
    }, [trick, conf]);

    

    if (true) {
      let percent = conf.toString() + "%"
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