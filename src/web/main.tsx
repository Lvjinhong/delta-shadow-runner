import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App.js";
import "./styles.css";
import "./map.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("缺少 Web 根节点 #root");
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
