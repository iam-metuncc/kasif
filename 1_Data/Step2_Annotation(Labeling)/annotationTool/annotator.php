<!DOCTYPE html>
<html lang="en">

<?php

// Include database connection and session settings
require 'config.php';

// Counter variable to track images in the slideshow
$i = 1;
$pageName = $_SESSION['pageName']; // Get current page name from session
$images = glob($_SESSION['currentDirectory'] . "*.png"); // Get all PNG images from the current directory (set in session)
$numberOfImages = count($images); // Count how many images we have

$annotator = $_SESSION['annotator'];
$_SESSION['numberOfTables'] = strval($numberOfImages); // Save number of tables (images) into session for later use

?>

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SnapTag - Annotate</title>
    <link href="css/style.css" rel="stylesheet">
    <!--<script type="text/javascript" src="js/script.js"></script>-->
</head>

<body>
    <!-- Sidebar panel with radio buttons for classes -->
    <div id="sidepanel">
        <div id="radioButtons">
            <form action="updateRecord.php" method="post" id="inputForm">
                <span class="entry">
                    <input type="radio" name="radio" id="col_zero" value="Column (0)" />
                    <label for="col_zero">Column (0)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="row_zero" value="Row (0)" />
                    <label for="row_zero">Row (0)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="row_zero_one" value="Row (0,1)" />
                    <label for="row_zero_one">Row (0,1)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="row_zero_one_two" value="Row (0,1,2)" />
                    <label for="row_zero_one_two">Row (0,1,2)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="value_attribute" value="Value Attribute" />
                    <label for="value_attribute">Value Attribute</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="row_zero_column_zero" value="Row (0), Column (0)" />
                    <label for="row_zero_column_zero">Row (0), Column (0)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="row_zero_one_column_zero" value="Row (0,1), Column (0)" />
                    <label for="row_zero_one_column_zero">Row (0,1), Column (0)</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="no_header" value="No header" />
                    <label for="no_header">No header</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="complex" value="Complex" />
                    <label for="complex">Complex</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="not_a_table" value="Not a table" />
                    <label for="not_a_table">Not a table</label>
                </span>

                <span class="entry">
                    <input type="radio" name="radio" id="others" value="Others" />
                    <label for="others">Others</label>
                </span>

                <span class="entry">
                    <?php
                    if ($numberOfImages > 0)
                        echo '<input type="submit" id="submitButton" name="annotate" value="Set Class">';
                    else
                        echo '<input type="submit" id="submitButton" name="nextPage" value="Next Page">';
                    ?>

                </span>

            </form>

            <?php
            if (isset($_POST['submit'])) {
                if (!empty($_POST['radio'])) {
                    echo '' . $_POST['radio'];
                } else {
                    echo 'Please select a class.';
                }
            }
            ?>

        </div>

        <p><a id="help" href="classes.html" target="_blank">Help</a></p>

    </div>

    <div id="mainContent">

        <div class="slideshow-container">


            <?php
            // $i will be used to keep track of how many images we have
            // $j will be used to access images
            $j = intval($_SESSION['tableNumber']);

            // Loop through all available images and display them
            while ($j <= $numberOfImages) {

                echo '<div class="mySlides">
                            <div class="numberText">' . $j . '/' . ($numberOfImages) . ', '. $_SESSION['pageName'] . '</div>
                            <img src="' . $images[$j - 1] . '">
                        </div>';
                $i = $i + 1;
                $j = $j + 1;
            }

            

            ?>

            <!-- Next and previous buttons -->
            <!--
            <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
            <a class="next" onclick="plusSlides(1)">&#10095;</a>
            -->
        </div>
    </div>
</body>


<!-- JavaScript for slideshow navigation -->
<script>
    let slideIndex = 1;
    showSlides(slideIndex);

    // Next/previous controls
    function plusSlides(n) {
        showSlides(slideIndex += n);
    }

    function showSlides(n) {
        let i;
        let slides = document.getElementsByClassName("mySlides");
        if (n > slides.length) {
            slideIndex = 1
        }
        if (n < 1) {
            slideIndex = slides.length
        }
        for (i = 0; i < slides.length; i++) {
            slides[i].style.display = "none";
        }
        slides[slideIndex - 1].style.display = "block";
    }
</script>

</html>



<!-- Slide show reference: https://www.w3schools.com/howto/howto_js_slideshow.asp -->