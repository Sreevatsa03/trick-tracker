import { useEffect } from "react";
import { useState } from "react";
import React from 'react';
import axios from 'axios';


const TrickDetected = () => {

    const [trick, setTrick] = useState(0)
    const [conf, setConf] = useState(0)

    useEffect(() => {
      // Make a GET request to the Flask API
      axios.get('http://127.0.0.1:8000/trick')
        .then((response) => {
          setTrick(response.data.Prediction);
          setConf(response.data.Accuracy);
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