import axios from "axios";

// Updated baseURL to point to your live Render backend
const API = axios.create({
    baseURL: "https://brain-hemorrhage-backend.onrender.com",
});

export const predictImage = async (imageFile) => {
    const formData = new FormData();
    formData.append("file", imageFile);

    const response = await API.post("/predict", formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });

    return response.data;
};

export default API;