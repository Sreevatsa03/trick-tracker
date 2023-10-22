import { useEffect } from "react";
import { useState } from "react";
import React from 'react';


const TrickDetected = () => {

    const [trick, setTrick] = useState(0)
    const [conf, setConf] = useState(0)

    useEffect(() => {
      // Make a GET request to the Flask API using the fetch API
      fetch('http://127.0.0.1:8000/trick?Accuracy=${conf}&Prediction=${trick}')
        .then((response) => {
          console.log(response);
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then((data) => {
          // Assuming the response is an object with 'Prediction' and 'Accuracy' properties
          setTrick(data.Prediction);
          setConf(data.Accuracy);
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