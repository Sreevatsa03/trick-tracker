import { useState, useEffect } from "react";
import React from 'react';
import axios from 'axios';


const TrickDetected = () => {

    const [trick, setTrick] = useState(0);
    const [conf, setConf] = useState(0);

    //const [data, setData] = useState(0);

    async function fetchData() {
      let response = await axios(
        `http://127.0.0.1:8000/trick`
      );
      let res = await response.data;
      setTrick(res["Prediction"]);
      setConf(res["Accuracy"]);
    }

    useEffect(() => {
      fetchData();
    }, []);

    
        

    //   fetch(`/trick?Accuracy=${conf}&Prediction=${trick}`)
    //     .then((response) => {
    //       if (!response.ok) {
    //         throw new Error('Network response was not ok');
    //       }
    //       return response.json();
    //     })
    //     .then((data) => {
    //       // Assuming the response is an object with 'Prediction' and 'Accuracy' properties
    //       setTrick(data.Prediction);
    //       setConf(data.Accuracy);
    //     })
    //     .catch((error) => {
    //       console.error('Error fetching data:', error);
    //     });
    // }, [trick, conf]);

    if (true) {
      let percent = conf
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