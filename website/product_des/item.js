document.addEventListener("DOMContentLoaded", function() {
    const quantityInput = document.querySelector('.qty-input');
    const increaseButton = document.querySelector('.plus-btn');
    const decreaseButton = document.getElementById('minus');
    var fadeTime = 300;
  
    // addEventListeners to check when user interacts w/ element
    increaseButton.addEventListener('click', function() {
      updateQtyBtn(1);
    });
  
    decreaseButton.addEventListener('click', function() {
      updateQtyBtn(-1);
    });
  
    quantityInput.addEventListener('change', function () {
      const newQuantity = parseInt(this.querySelector('.input-num').value, 10);
    });

    function updateQtyBtn(qtyInput) {
        /**
         * update the value of the input when pressing the increase decrease buttons
         * 
         * input:
         *  -qtyInput: which is the quantity that it should increase by
         */
        let currentQuantityInput = document.querySelector('.input-num');
        let currentQuantity = parseInt(currentQuantityInput.value, 10);
        currentQuantity += qtyInput;

        // Ensure quantity is greater than 0
        currentQuantity = Math.max(1, currentQuantity);

        currentQuantityInput.value = currentQuantity;
    }
});