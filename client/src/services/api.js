import axios from 'axios';

const API_BASE_URL = "https://brain-hemorrhage-backend.onrender.com";

export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export default {
  predictImage,
};