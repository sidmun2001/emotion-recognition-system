import { useState, useRef } from 'react';
import './App.css';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Header from './Header';




function Home() 
{
  const videoRef = useRef(null);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null); // State to store the uploaded file
  const navigate = useNavigate();
  


  // Start the camera stream
  const startCamera = async () => 
  {
    try 
    {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setCameraStarted(true);
    } 
    
    catch (error) {
      console.error('Error accessing the camera:', error);
    }
  };



  // Capture the image from the camera and upload it
  const captureAndUpload = async () => 
  {
    try 
    {
      if (!cameraStarted) 
      {
        alert('Please start the camera first!');
        return;
      }

      const canvas = document.createElement('canvas');
      const video = videoRef.current;

      // Set canvas dimensions to match the video stream
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to Blob
      const imageBlob = await new Promise
      ((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg')
      );

      const formData = new FormData();
      //const uniqueFilename = 'captured_image_' + Date.now() + '.jpg';
      formData.append('image', imageBlob, 'captured_image.jpg');
      
      const response = await axios.post
      ('http://localhost:8080/upload', formData, 
        {
          headers: 
          {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Image uploaded successfully:', response.data);

      // Navigate to the result page with response data
      const { uploadedImageUrl, preprocessedImageUrl, emotion } = response.data;
      navigate
      ('/result', 
        {
          state: 
          {
            uploadedImageUrl,
            preprocessedImageUrl,
            emotion,
          },
        }
      );
    } 
    

    catch (error) 
    {
      console.error('Error capturing or uploading image:', error);
    }
  };


  // Handle file selection
  const handleFileChange = (event) => 
  {
    setSelectedFile(event.target.files[0]);
  };



  // Upload the selected file
  const uploadImage = async () => 
  {
    try 
    {
      if (!selectedFile) 
      {
        alert('Please choose a file before uploading.');
        return;
      }

      const formData = new FormData();

      formData.append('image', selectedFile, 'captured_image.jpg');

      const response = await axios.post
      ('http://localhost:8080/upload', formData, 
        {
          headers: 
          {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Image uploaded successfully:', response.data);

      // Navigate to the result page with response data
      const 
      { uploadedImageUrl, preprocessedImageUrl, emotion } = response.data;
        navigate('/result', 
        {
          state: 
          {
            uploadedImageUrl,
            preprocessedImageUrl,
            emotion,
          },
          key: Date.now(), // Add a unique key to force re-mounting the component
        }
      );
    } 
    

    catch (error) 
    {
      console.error('Error uploading image:', error);
    }
  };



  
  return (
    <>
      <div className="App">
        <Header />
        <div className="container">
          {/* Video preview */}
          <video ref={videoRef} autoPlay className="video-preview"></video>
          {!cameraStarted && (
            <button onClick={startCamera} className="start-camera-btn">
              Start Camera
            </button>
          )}


          {/* Capture and Upload button appears when the camera is started */}
          {cameraStarted && (
            <button onClick={captureAndUpload} className="capture-upload-btn">
              Capture and Upload
            </button>
          )}
        </div>  
      </div>
    </>
  );
}



export default Home;