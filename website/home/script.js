//there is a basic fade in transition within general.js as place holder,
//feel free to unlink general.js from home.html when u put the actual animation in
//or alternatively u can leave the general animation in and play around with the cool animation
//for the main photo part that ppl see. :)

document.addEventListener('scroll', onScroll);

function onScroll () {
    var scrollPosition = window.scrollY,
        showHeaderPosition = 100;

    // Determine if position is at a certain point
    if (scrollPosition >= showHeaderPosition) {
        showHeader();
    } else {
        hideHeader();
    }
}

//https://markgoodyear.com/labs/headhesive/