// https://codepen.io/justinklemm/pen/kyMjjv

document.addEventListener("DOMContentLoaded", function() {
    const quantityInput = document.getElementById('inputNum');
    const increaseButton = document.querySelector('.plus-btn');
    const decreaseButton = document.querySelector('.minus-btn');

    increaseButton.addEventListener('click', function() {
      updateQuantity(1);
    });

    decreaseButton.addEventListener('click', function() {
      updateQuantity(-1);
    });

    function updateQuantity(change) {
      let currentQuantity = parseInt(quantityInput.value, 10);
      currentQuantity += change;

      // Ensure quantity is not negative
      currentQuantity = Math.max(1, currentQuantity);

      quantityInput.value = currentQuantity;
    }
  });