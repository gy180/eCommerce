<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);
$name = $email = $subject = $message = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Accessing form data using the $_POST superglobal
    $name = $_POST["name"];
    $email = $_POST["email"];
    $subject = $_POST["subject"];
    $message = $_POST["message"];

    $bodytxt = "Contacter Name: $name\n Email: $email\n\n Message:\n $message";
    $to = "hamitreestudio@gmail.com";

    $subjectHeader = "Message via Contact Form: $subject";

    $success = mail($to, $subject, $bodytxt);

    if($success){
        header("Location: form-sent.html");
        exit();
    }
    else {
        // Redirect to an error page if the email fails to send
        header("Location: form-error.html");
        exit();
    }

} else {
    header("Location: contact.html");
    exit();
}
?>
