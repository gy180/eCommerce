// https://codepen.io/justinklemm/pen/kyMjjv

document.addEventListener("DOMContentLoaded", function() {
  const quantityInput = document.querySelectorAll('.qty-input');
  const increaseButton = document.querySelectorAll('.plus-btn');
  const decreaseButton = document.querySelectorAll('.minus-btn');
  const removeButton = document.querySelectorAll('.remove-btn')
  var fadeTime = 300;
  recalculateCart();

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

  removeButton.forEach(function(button) {
    button.addEventListener('click', function() {
      const itemToRemove = this.closest('.cart-item');
      console.log(itemToRemove);
      itemToRemove.remove();
      recalculateCart();
    })
    
  })

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
    recalculateCart();
  }

  function recalculateCart(){
    const quantityInput = document.querySelectorAll('.cost-item-total');
    const discount = document.querySelector('.discount');
    const discountCost = discount.querySelector('.money-cost');
    let total = 0.0;
    quantityInput.forEach((element) => {
      total = total + parseFloat(element.textContent.substring(1));
    })
    const itemTotal = document.querySelector('.item-total');
    const itemCost = itemTotal.querySelector('.money-cost');
    itemCost.innerHTML = "$" + total.toFixed(2);
    const subtotal = document.querySelector('.subtotal');
    const subtotalCost = subtotal.querySelector('.money-cost');
    let value = total.toFixed(2) - parseFloat(discountCost.textContent.substring(1)).toFixed(2);
    subtotalCost.innerHTML = "$" + value.toFixed(2);
  }

  });


