// animation basics
document.addEventListener('DOMContentLoaded', function() {
    document.body.style.opacity = 1; // Set opacity to 1 to trigger the fade-in animation
    document.querySelector('header').style.opacity = 1; // Set opacity for the header
});

document.getElementById('navigation').style.display = 'blck';
document.getElementById('searchBar').style.display = 'none';

// search bar and navigation swap
document.getElementById('searchButton').addEventListener('click', function() {
    document.getElementById('navigation').style.display = 'none';
    document.getElementById('searchBar').style.display = 'block';
});

document.getElementById('xOut').addEventListener('click', function() {
    document.getElementById('navigation').style.display = 'block';
    document.getElementById('searchBar').style.display = 'none';
});