<?php

require 'config.php';

//Start clock
if(isset($_POST['annotate'])){
	//Registration form values

    $pageName = $_SESSION['pageName'];

    $tableName = $_SESSION['tableName'];

    // $tableName = 'table' . $tableNumber;

    $annotator = 'label' . $_SESSION['annotator'];

    $annotator_number = $_SESSION['annotator'];

    $label = strip_tags($_POST['radio']); //Remove html tags
	// $label = str_replace(' ', '', $label); //remove spaces
	// $label = ucfirst(strtolower($label)); //Uppercase first letter


    if ($annotator_number == 1 || $annotator_number == 2 || $annotator_number == 3)
         //Preparing the query string
        $queryString = 'UPDATE annotations SET '. $annotator . '="' . $label . '" WHERE pageName="' . $pageName . '" AND tableNumber="' . $tableName .'"';
    else
        $queryString = 'UPDATE annotations1 SET '. $annotator . '="' . $label . '" WHERE pageName="' . $pageName . '" AND tableNumber="' . $tableName .'"';
    
    $query = mysqli_query($con, $queryString);

    if(!$query){
        echo "Error (updating records): " . mysqli_error($con);
    }
    else{
        $tableNumberVal = intval($_SESSION['tableNumber']) + 1;
        $_SESSION['tableNumber'] = strval($tableNumberVal);
        $_SESSION['tableName'] =  'table' . strval($tableNumberVal);

    }

    //If we still have images to annotate then go back to the annotation UI
    if($tableNumberVal <= intval($_SESSION['numberOfTables'])){
        //update the annotator file (to keep track of where we are at)
        file_put_contents($_SESSION['annotatorFile'], $_SESSION['pageName'] . ' ' . $_SESSION['tableNumber']);
        header("Location: annotator.php");
    }
    else{
        header("Location: index.php?annotator=".$_SESSION['annotator'].'&nextPage=1');
    }


}

//End
exit();


?>
