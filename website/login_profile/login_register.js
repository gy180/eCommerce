

document.getElementById('loginPage').style.display = 'block';
document.getElementById('registerPage').style.display = 'none';

document.getElementById('changeToRegister').addEventListener('click', function() {
    document.getElementById('loginPage').style.display = 'none';
    document.getElementById('registerPage').style.display = 'block';
});


document.getElementById('changeToLogin').addEventListener('click', function() {
    document.getElementById('loginPage').style.display = 'block';
    document.getElementById('registerPage').style.display = 'none';
});