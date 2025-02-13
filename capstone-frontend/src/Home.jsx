import React, { useState, useRef } from 'react';
import './App.css';
import axios from 'axios';
import Header from './Header';
import EmotionPieChart from "./EmotionPieChart";




function Home() {
  const videoRef = useRef(null);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [emotionData, setEmotionData] = useState(null);



  const startCamera = async () => 
  {
    try 
    {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setCameraStarted(true);
    } 
    
    catch (error) 
    {
      console.error('Error accessing the camera:', error);
    }
  };



  const captureAndUpload = async () => 
  {
    if (!cameraStarted) 
    {
      alert('Please start the camera first!');
      return;
    }


    const canvas = document.createElement('canvas');
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);


    const imageBlob = await new Promise
    (
      (resolve) =>
      canvas.toBlob(resolve, 'image/jpeg')
    );


    const formData = new FormData();
    formData.append('image', imageBlob, 'captured_image.jpg');


    try 
    {
      const response = await axios.post
      ('http://localhost:8080/upload', formData, 
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );


      setCapturedImage(response.data.uploadedImageUrl);
      setPreprocessedImage(response.data.preprocessedImageUrl);
      setEmotionData(response.data.emotionData);
    } 
    
    catch (error) 
    {
      console.error('Error uploading image:', error);
    }
  };



  const getDetectedEmotion = () => 
  {
    if (!emotionData) return null;
  
    const sortedEmotions = Object.entries(emotionData).sort((a, b) => b[1] - a[1]);
    return sortedEmotions[0][0]; // Get emotion with the highest probability
  };
  


  const barChartData = 
  {
    labels: emotionData ? Object.keys(emotionData) : [],
    datasets: 
    [
      {
        label: 'Emotion Probabilities',
        data: emotionData ? Object.values(emotionData) : [],
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
      },
    ],
  };



  return (
    <div className="App">


      <Header />
      <div className="main-container">
        <div className="camera-section">
          <video ref={videoRef} autoPlay className="video-preview"></video>
          {!cameraStarted && (
            <button onClick={startCamera} className="start-camera-btn">
              Start Camera
            </button>
          )}
          {cameraStarted && (
            <button onClick={captureAndUpload} className="capture-upload-btn">
              Capture and Upload
            </button>
          )}
        </div>

        <div className="results-section">
          <div className="image-box">
            <h3>Captured Photo</h3>
            {capturedImage ? (
              <img src={capturedImage + "?t=" + new Date().getTime()} alt="Captured" className="result-image" />
            ) : (
              <div className="placeholder-box">No Image</div>
            )} 
          </div>

          <div className="image-box">
            <h3>Preprocessed Photo</h3>
            {preprocessedImage ? (
                <img src={preprocessedImage + "?t=" + new Date().getTime()} alt="Preprocessed" className="result-image" />
              ) : (
                <div className="placeholder-box">No Image</div>
              )}
          </div>
        </div>


        <div className="detected-emotion">
          <h2>Detected Emotion: <span className="emotion-text">{getDetectedEmotion() || "No Data"}</span></h2>
        </div>


        <div className="chart-container">
          <h3 className="chart-title">Emotion Probabilities</h3>
          {emotionData ? <EmotionPieChart emotionData={emotionData} /> : <div className="placeholder-box">No Data</div>}
        </div>


      </div>
    </div>
  );
}



export default Home;