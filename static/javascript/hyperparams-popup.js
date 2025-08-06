document.addEventListener("DOMContentLoaded", () => {
  const openParamsBtn = document.getElementById("showParamsBtn");
  const paramsModal = document.getElementById("paramsModal");
  const closePopupBtn = paramsModal.querySelector(".close-btn");

  if (!openParamsBtn || !paramsModal || !closePopupBtn) {
    console.error("One or more modal elements are missing.");
    return;
  }

  openParamsBtn.addEventListener("click", () => {
    paramsModal.style.display = "block";
  });

  closePopupBtn.addEventListener("click", () => {
    paramsModal.style.display = "none";
  });

  window.addEventListener("click", (event) => {
    if (event.target === paramsModal) {
      paramsModal.style.display = "none";
    }
  });
});