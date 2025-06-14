import { useEffect, useRef, useState } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import * as tf from "@tensorflow/tfjs";
const LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
const NOSE = [1, 2, 98, 327];
const REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173];
const LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398];
const LHAND = Array.from({ length: 21 }, (_, i) => 468 + i);
const RHAND = Array.from({ length: 21 }, (_, i) => 522 + i);

function App() {
  const videoRef = useRef(null);
  const [prediction, setPrediction] = useState("...");
  const [labelMap, setLabelMap] = useState({});

  useEffect(() => {
    fetch("/sign_to_prediction_index_map.json")
      .then((res) => res.json())
      .then((data) => {
        const reverseMap = {};
        for (const [label, index] of Object.entries(data)) {
          reverseMap[index] = label;
        }
        setLabelMap(reverseMap);
      });
  }, []);

  useEffect(() => {
    if (!videoRef.current) return;

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults(onResults);

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    camera.start();

    function onResults(results) {
      const sequence = extractPoints(results); // Shape: [384, 708]
      if (!sequence) return;

      fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence }),
      })
        .then((res) => res.json())
        .then((json) => {
          if (json.prediction !== undefined) {
            const word = labelMap[json.prediction] || `Unknown (${json.prediction})`;
            setPrediction(word);
          } else {
            setPrediction("No prediction");
          }
        })
        .catch(() => setPrediction("API error"));
    }

    // TODO: Stop camera/holistic on unmount
    return () => holistic.close();
  }, [labelMap]);

  function extractPoints(results) {
    // Collect required landmarks and flatten into shape [384, 708]
    let allPoints = [];

    const extractXYZ = (lm, i) => lm?.[i] ? [lm[i].x, lm[i].y, lm[i].z] : [NaN, NaN, NaN];

    const collect = (lm, indices) => indices.map(i => extractXYZ(lm, i)).flat();

    const face = collect(results.faceLandmarks, [...LIP, ...REYE, ...LEYE]);
    const leftHand = collect(results.leftHandLandmarks, LHAND);
    const rightHand = collect(results.rightHandLandmarks, RHAND);
    const nose = collect(results.faceLandmarks, NOSE);

    allPoints = [...face, ...leftHand, ...rightHand, ...nose];

    if (allPoints.length !== 708) return null;

    return Array.from({ length: 384 }, () => allPoints); // replicate for dummy time sequence
  }

  return (
    <div className="bg-black text-white min-h-screen flex flex-col items-center justify-center">
      <video ref={videoRef} autoPlay muted playsInline className="w-full max-w-xl" />
      <h2 className="text-xl mt-4 text-yellow-400">Prediction:</h2>
      <div className="text-4xl font-bold">{prediction}</div>
    </div>
  );
}

export default App;
