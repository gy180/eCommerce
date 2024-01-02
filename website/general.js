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

function redirectPage(pagePath){
    /**
     * On button click, page gets redirected
     * Input: pagePath - the relative path that it'll get redirected to
     */
    window.location.href = pagePath
}

//sign in ? https://uiverse.io/Yaya12085/short-panda-24?username=&password=
// https://www.w3schools.com/howto/howto_js_popup_form.asp
