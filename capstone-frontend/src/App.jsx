import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './home';
import ResultPage from './ResultPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/result" element={<ResultPage/>} />
      </Routes>
    </Router>
  );
}

export default App;