import { useEffect, useRef, useState } from "react";
import { Holistic, Results as HolisticResults } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import "@mediapipe/holistic";

const DESIRED_FRAME_COUNT = 200;

const filteredIndices = {
  pose: [11, 12, 13, 14, 15, 16],
  hands: Array.from({ length: 21 }, (_, i) => i),
  face: [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61,
    63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109,
    127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158,
    159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251,
    263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300,
    308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361,
    362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388,
    389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 473]
};

function App() {
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [requestPayload, setRequestPayload] = useState<any>(null);

  const [landmarkFrames, setLandmarkFrames] = useState<number[][][]>([]);

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
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

    const camera = new Camera(videoRef.current!, {
      onFrame: async () => {
        await holistic.send({ image: videoRef.current! });
      },
      width: 640,
      height: 480,
    });

    camera.start();

    function onResults(results: HolisticResults) {
      const ctx = canvasRef.current!.getContext("2d")!;
      ctx.clearRect(0, 0, 640, 480);
      ctx.drawImage(results.image, 0, 0, 640, 480);

      const allLandmarks: number[][] = [];

      // Pose
      if (results.poseLandmarks) {
        filteredIndices.pose.forEach(i => {
          const lm = results.poseLandmarks![i];
          allLandmarks.push([lm.x, lm.y, lm.z]);
        });
      }

      // Hands
      if (results.leftHandLandmarks) {
        filteredIndices.hands.forEach(i => {
          const lm = results.leftHandLandmarks![i];
          allLandmarks.push([lm.x, lm.y, lm.z]);
        });
      } else {
        allLandmarks.push(...Array(21).fill([0, 0, 0]));
      }

      if (results.rightHandLandmarks) {
        filteredIndices.hands.forEach(i => {
          const lm = results.rightHandLandmarks![i];
          allLandmarks.push([lm.x, lm.y, lm.z]);
        });
      } else {
        allLandmarks.push(...Array(21).fill([0, 0, 0]));
      }

      // Face
      if (results.faceLandmarks) {
        filteredIndices.face.forEach(i => {
          const lm = results.faceLandmarks![i];
          allLandmarks.push([lm.x, lm.y, lm.z]);
        });
      } else {
        allLandmarks.push(...Array(filteredIndices.face.length).fill([0, 0, 0]));
      }

      // Draw on canvas
      ctx.fillStyle = "lime";
      allLandmarks.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * 640, y * 480, 2, 0, 2 * Math.PI);
        ctx.fill();
      });

      // Accumulate 200 frames
      setLandmarkFrames(prev => {
        const updated = [...prev, allLandmarks];
        if (updated.length >= DESIRED_FRAME_COUNT) {
          sendToBackend(updated.slice(0, DESIRED_FRAME_COUNT));
          return [];
        }
        return updated;
      });
    }

    return () => {
      camera.stop();
    };
  }, []);

  const sendToBackend = async (landmarks: number[][][]) => {
  try {
    setRequestPayload(landmarks);  // <-- Store what is being sent
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks }),
    });
    const data = await res.json();
    console.log("Prediction:", data.prediction);
    setPredictionResult(data);  // <-- store the result in state
  } catch (err) {
    console.error("Error:", err);
    setPredictionResult({ error: "Failed to get prediction." });
  }
};


  return (
    <div className="flex flex-col items-center p-4">
      <div className="relative w-[640px] h-[480px]">
        <video ref={videoRef} className="absolute" width="640" height="480" autoPlay muted />
        <canvas ref={canvasRef} className="absolute" width="640" height="480" />
      </div>
      <p className="mt-4 text-gray-600 text-sm">
        Recording 200 frames of filtered landmarks for inference...
      </p>
      {predictionResult && (
      <pre className="mt-4 w-full max-w-[640px] bg-gray-100 p-2 rounded text-xs overflow-x-auto">
        {JSON.stringify(predictionResult, null, 2)}
      </pre>)}
      {requestPayload && (
  <pre className="mt-4 w-full max-w-[640px] bg-yellow-50 p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
    <strong className="block font-semibold mb-1">Request Payload Summary:</strong>
    {JSON.stringify(
      {
        totalFrames: requestPayload.length,
        pointsPerFrame: requestPayload[0]?.length || 0,
        firstFrameSample: requestPayload[0]?.slice(0, 200), // show first 5 points only
      },
      null,
      2
    )}
  </pre>
)}

    </div>
  );
}

export default App;