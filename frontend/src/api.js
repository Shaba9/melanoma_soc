import axios from "axios";

const baseURL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({ baseURL });

export async function predict(file, model = "augmented") {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post(`/predict?model=${model}`, form);
  return data;
}

export async function heatmap(file, model = "augmented") {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post(`/heatmap?model=${model}`, form, {
    responseType: "blob",
  });
  return URL.createObjectURL(res.data);
}

export async function getMetrics(exp = "augmented") {
  const { data } = await client.get(`/metrics?exp=${exp}`);
  return data;
}

export async function getExperiments() {
  const { data } = await client.get("/experiments");
  return data;
}

export async function getFigures() {
  const { data } = await client.get("/figures");
  return data.figures;
}

export function figureUrl(name) {
  return `${baseURL}/figures/${name}`;
}

export default client;
