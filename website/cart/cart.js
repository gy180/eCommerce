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

  /**
   *  document.addEventListener("DOMContentLoaded", function() {
      const items = [
        { id: 1, name: 'Item 1', price: 10, quantity: 1 },
        { id: 2, name: 'Item 2', price: 15, quantity: 1 },
        { id: 3, name: 'Item 3', price: 20, quantity: 1 }
      ];

      const cartContainer = document.getElementById('cart-container');
      const couponInput = document.getElementById('coupon-input');
      const applyButton = document.getElementById('apply-btn');
      const totalCostElement = document.getElementById('total-cost');

      // Render initial cart items
      renderCart();

      applyButton.addEventListener('click', function() {
        applyCoupon();
      });

      function renderCart() {
        cartContainer.innerHTML = '';

        items.forEach(item => {
          const itemElement = document.createElement('div');
          itemElement.classList.add('cart-item');

          const itemName = document.createElement('span');
          itemName.textContent = item.name;

          const qtyContainer = document.createElement('div');
          qtyContainer.classList.add('qty-container');

          const decreaseBtn = document.createElement('button');
          decreaseBtn.textContent = '-';
          decreaseBtn.addEventListener('click', function() {
            updateQuantity(item.id, -1);
          });

          const qtyInput = document.createElement('input');
          qtyInput.classList.add('qty-input');
          qtyInput.type = 'text';
          qtyInput.value = item.quantity;
          qtyInput.readOnly = true;

          const increaseBtn = document.createElement('button');
          increaseBtn.textContent = '+';
          increaseBtn.addEventListener('click', function() {
            updateQuantity(item.id, 1);
          });

          const removeBtn = document.createElement('button');
          removeBtn.textContent = 'Remove';
          removeBtn.addEventListener('click', function() {
            removeItem(item.id);
          });

          qtyContainer.appendChild(decreaseBtn);
          qtyContainer.appendChild(qtyInput);
          qtyContainer.appendChild(increaseBtn);

          itemElement.appendChild(itemName);
          itemElement.appendChild(qtyContainer);
          itemElement.appendChild(removeBtn);

          cartContainer.appendChild(itemElement);
        });

        updateTotalCost();
      }

      function updateQuantity(itemId, change) {
        const item = items.find(item => item.id === itemId);

        if (item) {
          item.quantity += change;
          item.quantity = Math.max(1, item.quantity);
          renderCart();
        }
      }

      function removeItem(itemId) {
        const index = items.findIndex(item => item.id === itemId);

        if (index !== -1) {
          items.splice(index, 1);
          renderCart();
        }
      }

      function applyCoupon() {
        const enteredCoupon = couponInput.value.trim();
        const validCouponCode = "SALE2023"; // Replace with your valid coupon code

        if (enteredCoupon === validCouponCode) {
          // Apply coupon discount or perform other coupon-related logic
          alert('Coupon applied successfully!');
        } else {
          alert('Invalid coupon code. Please try again.');
        }
      }

      function updateTotalCost() {
        const totalCost = items.reduce((sum, item) => sum + item.price * item.quantity, 0);
        totalCostElement.textContent = `Total Cost: $${totalCost}`;
      }
    });
   */


