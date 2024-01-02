//there is a basic fade in transition within general.js as place holder,
//feel free to unlink general.js from home.html when u put the actual animation in
//or alternatively u can leave the general animation in and play around with the cool animation
//for the main photo part that ppl see. :)

document.addEventListener("DOMContentLoaded", function() {
    var navbar = document.getElementById('main-header');
    const triggerPos = document.getElementById('menu').getBoundingClientRect().top;
    navbar.style.opacity = "0";
    window.onscroll = function() {
        // Check if the user has scrolled down (you can adjust the threshold as needed)
        if (window.scrollY > triggerPos) {
            navbar.style.opacity = "1";
        } else {
            navbar.style.opacity = "0";
        }
    };
});

//https://markgoodyear.com/labs/headhesive/