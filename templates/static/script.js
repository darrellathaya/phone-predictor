const ramRange = document.getElementById("ramRange");
const ramInput = document.getElementById("ramInput");
ramRange.addEventListener("input", () => ramInput.value = ramRange.value);
ramInput.addEventListener("input", () => ramRange.value = ramInput.value);

const storageRange = document.getElementById("storageRange");
const storageInput = document.getElementById("storageInput");
storageRange.addEventListener("input", () => storageInput.value = storageRange.value);
storageInput.addEventListener("input", () => storageRange.value = storageInput.value);

document.querySelector("form").addEventListener("submit", function() {
  document.getElementById("submitBtn").disabled = true;
  document.getElementById("loadingSpinner").classList.remove("d-none");
});

// Add this after your page loads
const element = document.querySelector('.form-select');
const choices = new Choices(element, {
  searchEnabled: true,
  shouldSort: false,
  position: 'bottom', // or 'top'
  maxItemCount: 1,
  classNames: {
    containerOuter: 'choices form-select'
  }
});