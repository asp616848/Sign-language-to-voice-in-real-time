import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./webSocket";
// App.tsx
function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { sendFrame } = useWebSocket(handlePrediction);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    });
  }, []);

  const captureAndSend = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (canvas && video) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx?.drawImage(video, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) sendFrame(blob);
      }, "image/jpeg");
    }
  };

  function handlePrediction(prediction: any) {
    console.log("Prediction:", prediction);
    // TODO: Draw prediction results
  }

  return (
    <div>
      <video ref={videoRef} autoPlay muted />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <button onClick={captureAndSend}>Send Frame</button>
    </div>
  );
}
