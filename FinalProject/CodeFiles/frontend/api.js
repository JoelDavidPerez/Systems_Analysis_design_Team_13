import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

export const trainModel = async (file, epochs = 50) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('epochs', epochs);
  
  const response = await axios.post(`${API_URL}/train`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  return response.data;
};

export const predictTest = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  return response.data;
};

export const loadModel = async () => {
  const response = await axios.get(`${API_URL}/load_model`);
  return response.data;
};