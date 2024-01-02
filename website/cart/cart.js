// https://codepen.io/justinklemm/pen/kyMjjv

document.addEventListener("DOMContentLoaded", function() {
  const quantityInput = document.querySelectorAll('.qty-input');
  const increaseButton = document.querySelectorAll('.plus-btn');
  const decreaseButton = document.querySelectorAll('.minus-btn');
  const removeButton = document.querySelectorAll('.remove-btn')
  var fadeTime = 300;

  // addEventListeners to check when user interacts w/ element
  increaseButton.forEach(function(button) {
    button.addEventListener('click', function() {
      updateQtyBtn(this, 1);
    });
  });

  decreaseButton.forEach(function(button) {
    button.addEventListener('click', function() {
      updateQtyBtn(this, -1);
    });
  });

  quantityInput.forEach(function(input) {
    input.addEventListener('change', function () {
      const newQuantity = parseInt(this.value, 10);
      const cartItem = this.closest('.cart-item');
      const costItem = cartItem.querySelector('.cost-item-total');

      updateItemCost(costItem, newQuantity);
    });
  });

  function updateQtyBtn(clickedButton, qtyInput) {
    /**
     * update the value of the input when pressing the increase decrease buttons
     * 
     * input:
     *  -qtyInput: which is the quantity that it should increase by
     */
    let currentQuantityInput = clickedButton.parentNode.querySelector('.qty-input');
    let currentQuantity = parseInt(currentQuantityInput.value, 10);
    currentQuantity += qtyInput;

    // Ensure quantity is greater than 0
    currentQuantity = Math.max(1, currentQuantity);

    currentQuantityInput.value = currentQuantity;
    const cartItem = clickedButton.closest('.cart-item');
    const costItem = cartItem.querySelector('.cost-item-total');
    updateItemCost(costItem, currentQuantity);
  }
  
  function updateItemCost(cost, qty) {
    /**
     * update the total item cost based on the quantity
     * 
     * inputs:
     *  -cost: the location of the cost to be updated
     *  - qty: the new quantity (an int)
     */
    let productCostTotal = 0;
    const perCost = cost.nextElementSibling.textContent;
    const perCostValue = parseFloat(perCost.slice(3,9));
    productCostTotal = qty * perCostValue;
    cost.innerHTML = "$" + productCostTotal.toFixed(2);
  }

  function recalculateCart(){

  }

  function removeItem(removeButton){
    
  }


  });


