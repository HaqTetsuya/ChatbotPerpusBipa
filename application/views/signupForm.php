<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/jquery.validation/1.19.5/jquery.validate.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <style>
        body {
            font-family: 'Patrick Hand', cursive;
            padding-top: 80px;
        }
    </style>
</head>

<body class="bg-white h-screen flex items-center justify-center">
    <div class="bg-white w-full max-w-md rounded-lg shadow-lg p-8 flex flex-col border-2 border-black mt-12">
		<?php if ($this->session->flashdata('errors')): ?>
			<ul>
				<?php foreach ($this->session->flashdata('errors') as $field => $error): ?>
					<li><strong><?= ucfirst($field) ?>:</strong> <?= $error ?></li>
				<?php endforeach; ?>
			</ul>
		<?php endif; ?>

        <div class="flex flex-col items-center mb-8">
            <div class="w-24 h-24 border-2 border-black rounded-full mb-4 flex items-center justify-center">
                <i class="fas fa-user-plus text-4xl"></i>
            </div>
            <h1 class="text-3xl font-bold">Sign Up</h1>
        </div>
		
        <form action="<?php echo base_url('auth/aksiSign') ?>" id="signup-form" class="flex flex-col space-y-4" method="POST">
            <div class="flex flex-col">
                <label for="email" class="mb-1 font-bold">Email</label>
                <input type="email" id="email" name="email" placeholder="Enter your email"
                    class="p-3 border-2 border-black rounded-lg" required>
            </div>

            <div class="flex flex-col">
                <label for="nama" class="mb-1 font-bold">nama</label>
                <input type="text" id="nama" name="nama" placeholder="Choose a name"
                    class="p-3 border-2 border-black rounded-lg" required>
            </div>

            <div class="flex flex-col">
                <label for="password" class="mb-1 font-bold">Password</label>
                <div class="relative">
                    <input type="password" id="password" name="password" placeholder="Create a password"
                        class="p-3 border-2 border-black rounded-lg w-full" required>
                    <button type="button" id="toggle-password" class="absolute right-3 top-3">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </div>

            <div class="flex flex-col">
                <label for="confirm-password" class="mb-1 font-bold">Confirm Password</label>
                <div class="relative">
                    <input type="password" id="confirm-password" name="confirm_password" placeholder="Confirm your password"
                        class="p-3 border-2 border-black rounded-lg w-full" required>
                    <button type="button" id="toggle-confirm-password" class="absolute right-3 top-3">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            </div>
            <!--
            <div class="flex items-center">
                <input type="checkbox" id="terms" name="terms" class="mr-2 h-4 w-4 border-2 border-black" required>
                <label for="terms">I agree to the <a href="#" class="font-bold underline">Terms and Conditions</a></label>
            </div>
            -->
            <button type="submit" class="bg-white border-2 border-black p-3 rounded-lg font-bold hover:bg-gray-100 transition duration-300">
                Create Account
            </button>
        </form>


        <div class="mt-6 text-center">
            <p>Already have an account? <a href="<?php echo base_url('auth/login') ?>" class="font-bold underline">Login</a></p>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            // Toggle password visibility
            $('#toggle-password').click(function() {
                const passwordInput = $('#password');
                const icon = $(this).find('i');

                if (passwordInput.attr('type') === 'password') {
                    passwordInput.attr('type', 'text');
                    icon.removeClass('fa-eye').addClass('fa-eye-slash');
                } else {
                    passwordInput.attr('type', 'password');
                    icon.removeClass('fa-eye-slash').addClass('fa-eye');
                }
            });

            // Toggle confirm password visibility
            $('#toggle-confirm-password').click(function() {
                const passwordInput = $('#confirm-password');
                const icon = $(this).find('i');

                if (passwordInput.attr('type') === 'password') {
                    passwordInput.attr('type', 'text');
                    icon.removeClass('fa-eye').addClass('fa-eye-slash');
                } else {
                    passwordInput.attr('type', 'password');
                    icon.removeClass('fa-eye-slash').addClass('fa-eye');
                }
            });
        });
    </script>

</body>

</html>