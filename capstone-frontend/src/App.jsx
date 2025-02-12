import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import ResultPage from './ResultPage';
import EmotionPieChart from './EmotionPieChart';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/result" element={<ResultPage/>} />
        <Route path="/chart" element={<EmotionPieChart/>} />
      </Routes>
    </Router>
  );
}

export default App;