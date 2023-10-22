import { useState } from "react";
import React from 'react';

const TrickDetected = () => {

    // const [trick, setTrick] = useState(null);
    // const [conf, setConf] = useState(null);

    const [data, setData] = useState({});
    const [visible, setVisible] = useState(false);


    const beginQuery = async() => {
      await fetch("http://127.0.0.1:8000/trick", {
        method: 'GET',
        headers: {
          accept: 'application/json',
        }
      }).then((res => res.json())).then(dataobject => {
        console.log(dataobject);
        setData(dataobject);
      })

      setVisible(true);

    }
        
    let conf = data["Accuracy"];
    let trick = data["Prediction"];
    let percent = parseFloat(conf)*100;
    percent = Math.round(percent * 10) / 10
    percent.toString();
    percent = percent + "%";

    return (
        <div>
          <button onClick={beginQuery}> Analyze trick </button>
          {visible ? <h2 id="analysis">{trick} detected with {percent} accuracy!</h2> : null}
        </div>
      );

  
};

export default TrickDetected;