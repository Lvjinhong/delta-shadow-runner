import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("缺少 Web 根节点 #root");
}

createRoot(rootElement).render(
  <StrictMode>
    <main>Shadow Runner Lab 项目基线</main>
  </StrictMode>,
);
