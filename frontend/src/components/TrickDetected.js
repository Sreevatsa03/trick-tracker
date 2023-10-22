import { useState, useEffect } from "react";
import React from 'react';


const TrickDetected = () => {

    // const [trick, setTrick] = useState(null);
    // const [conf, setConf] = useState(null);

    const [data, setData] = useState({});


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
    }
<<<<<<< HEAD

    useEffect(() => {
      fetchData();
    }, []);

    
=======
  
>>>>>>> 36012241 (more api debugging)
        

    if (true) {
      return (
          <div>
            
            <button onClick={ () => beginQuery() }>Make get request</button>
            <h2>{data["Prediction"]} detected with {data["Accuracy"]} accuracy!</h2>
          </div>
        );
    } else {
        return (<div></div>);
    }

  
};

export default TrickDetected;