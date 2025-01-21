import { useState, useEffect, useRef } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { use } from 'react'
import axios from 'axios'

function App() {
  const [count, setCount] = useState(0)
  const [users, setUsers] = useState([]);

  const fetchAPI = async () => {
    const response = await axios.get("http://localhost:8080/api/users");
    console.log(response.data.users);
    setUsers(response.data.users);
  }

  const videoRef = useRef(null);
  const [cameraStarted, setCameraStarted] = useState(false);

  // Start the camera stream
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setCameraStarted(true);
    } catch (error) {
      console.error('Error accessing the camera:', error);
    }
  };

  // Capture the image and upload
  const captureAndUpload = async () => {
    try {
      const canvas = document.createElement('canvas');
      const video = videoRef.current;

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');

      // Draw the current video frame onto the canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to Blob
      const imageBlob = await new Promise((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg')
      );

      // Prepare FormData for Axios request
      const formData = new FormData();
      formData.append('image', imageBlob, 'captured_image.jpg');

      // Send the image to the backend
      const response = await axios.post('http://localhost:8080/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Image uploaded successfully:', response.data);
    } catch (error) {
      console.error('Error capturing or uploading image:', error);
    }
  };

  useEffect(() => { 
    fetchAPI();
  }, []);

  
  return (
    <>
      <header class="header-bar">
        <div class="dropdown">
            <button class="dropbtn">☰</button>
            <div class="dropdown-content">
                <a href="/restart">Restart</a>
                <a href="/delete">Delete</a>
                <a href="/close">Close</a>
            </div>
        </div>
        <h1 class="header-title">Emotion Recognition System</h1>
    </header>
       <div>
      <video ref={videoRef} autoPlay style={{ width: '100%' }}></video>
      {!cameraStarted && <button onClick={startCamera}>Start Camera</button>}
      {cameraStarted && (
        <button onClick={captureAndUpload}>Capture and Upload</button>
      )}
    </div>
      <div className="card card-1">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          {users.map((user, index) => ( 
            <div key={index}>
              <span>{user.name}</span>
              <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
            </div>
          ))}
        </p>
      </div>
      <div class="container">
        <h1>Emotion Recognition</h1>

        {/* <!-- Option to upload an image from the computer --> */}
        <form action="/result" method="POST" enctype="multipart/form-data">
            <label for="imageUpload">Choose an image from your computer:</label>
            <input type="file" id="imageUpload" name="image" accept="image/*"></input>
            <button type="submit" class="button">Analyze</button>
        </form>
    </div>
    </>
  )
}

export default App
