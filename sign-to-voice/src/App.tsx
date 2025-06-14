import React, { useRef, useEffect, useState } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import * as cam from "@mediapipe/camera_utils";

const filteredHand = Array.from({ length: 21 }, (_, i) => i);
const filteredPose = [11, 12, 13, 14, 15, 16];
const filteredFace = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
  61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
  107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
  155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
  234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
  293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
  332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
  381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
  415, 454, 466, 468, 473];

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [landmarkData, setLandmarkData] = useState<number[][]>([]);

  useEffect(() => {
    const holistic = new Holistic({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    holistic.onResults((results) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const video = videoRef.current;

      if (!ctx || !canvas || !video) return;

      // Resize canvas to video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const allLandmarks: number[][] = [];

      const drawPoints = (landmarks: any[], filter: number[]) => {
        filter.forEach((i) => {
          if (landmarks[i]) {
            const { x, y, z } = landmarks[i];
            const cx = x * canvas.width;
            const cy = y * canvas.height;
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
            ctx.fillStyle = "red";
            ctx.fill();
            allLandmarks.push([x, y, z]);
          }
        });
      };

      if (results.poseLandmarks)
        drawPoints(results.poseLandmarks, filteredPose);
      if (results.faceLandmarks)
        drawPoints(results.faceLandmarks, filteredFace);
      if (results.leftHandLandmarks)
        drawPoints(results.leftHandLandmarks, filteredHand);
      if (results.rightHandLandmarks)
        drawPoints(results.rightHandLandmarks, filteredHand);

      setLandmarkData(allLandmarks); // Can be sent to backend via API
    });

    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          await holistic.send({ image: videoRef.current! });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }

    return () => {
      holistic.close();
    };
  }, []);

  return (
    <div className="w-fit mx-auto mt-4">
      <video ref={videoRef} className="absolute top-0 left-0 rounded-xl shadow-lg" autoPlay playsInline muted />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0"
        style={{ zIndex: 10 }}
      />
      <pre className="mt-4 max-h-48 overflow-y-auto bg-gray-900 text-green-400 p-2 text-xs">
        {JSON.stringify(landmarkData, null, 2)}
      </pre>
    </div>
  );
};

export default App;
